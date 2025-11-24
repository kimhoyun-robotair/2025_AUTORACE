#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <vector>
#include <deque>
#include <iostream>
#include <utility>   // std::pair
#include <iomanip>   // std::setprecision

struct Point2D
{
  double x;
  double y;
};

class ScanProcessor : public rclcpp::Node
{
public:
  ScanProcessor()
  : Node("scan_processor"),
    window_size_(20),
    merge_threshold_(0.18),
    speed_window_size_(10),
    speed_swap_margin_(0.02),   // 평균 속도 차이가 2cm/frame 이상일 때만 swap
    swap_cooldown_(0),
    swap_cooldown_max_(15)      // swap 후 15프레임 동안은 다시 swap 금지
  {
    // 기본 필터 파라미터
    this->declare_parameter<double>("scan_range_min", 0.0);
    this->declare_parameter<double>("scan_range_max", 10.0);
    this->declare_parameter<double>("scan_angle_min", -M_PI / 3.0);
    this->declare_parameter<double>("scan_angle_max",  M_PI / 3.0);

    this->get_parameter("scan_range_min",  scan_range_min_);
    this->get_parameter("scan_range_max",  scan_range_max_);
    this->get_parameter("scan_angle_min",  scan_angle_min_);
    this->get_parameter("scan_angle_max",  scan_angle_max_);

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ScanProcessor::scanCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "cluster_centers", 10);
  }

private:
  // ---------- 유틸 함수들 ----------

  double dist(const Point2D &a, const Point2D &b) const
  {
    return std::hypot(a.x - b.x, a.y - b.y);
  }

  Point2D add(const Point2D &a, const Point2D &b) const
  {
    return {a.x + b.x, a.y + b.y};
  }

  Point2D sub(const Point2D &a, const Point2D &b) const
  {
    return {a.x - b.x, a.y - b.y};
  }

  // points 에 대해 K-means(k=2)
  std::pair<Point2D, Point2D> kmeans2(const std::vector<Point2D> &pts) const
  {
    // 최소 2개는 있다고 가정
    Point2D c1 = pts.front();
    Point2D c2 = pts.back();

    const int max_iter = 10;
    std::vector<int> labels(pts.size(), 0);

    for (int iter = 0; iter < max_iter; ++iter)
    {
      // 1) 할당
      for (size_t i = 0; i < pts.size(); ++i)
      {
        double d1 = dist(pts[i], c1);
        double d2 = dist(pts[i], c2);
        labels[i] = (d1 <= d2) ? 0 : 1;
      }

      // 2) 새로운 중심 계산
      Point2D new_c1{0.0, 0.0}, new_c2{0.0, 0.0};
      int count1 = 0, count2 = 0;

      for (size_t i = 0; i < pts.size(); ++i)
      {
        if (labels[i] == 0)
        {
          new_c1.x += pts[i].x;
          new_c1.y += pts[i].y;
          ++count1;
        }
        else
        {
          new_c2.x += pts[i].x;
          new_c2.y += pts[i].y;
          ++count2;
        }
      }

      if (count1 > 0)
      {
        new_c1.x /= static_cast<double>(count1);
        new_c1.y /= static_cast<double>(count1);
      }
      if (count2 > 0)
      {
        new_c2.x /= static_cast<double>(count2);
        new_c2.y /= static_cast<double>(count2);
      }

      double move1 = dist(c1, new_c1);
      double move2 = dist(c2, new_c2);
      c1 = new_c1;
      c2 = new_c2;
      if (move1 < 1e-4 && move2 < 1e-4)
        break;
    }

    return {c1, c2};
  }

  // deque 에 새 점을 넣고, 평균을 반환 (smoothing)
  Point2D pushAndSmooth(std::deque<Point2D> &hist, const Point2D &p)
  {
    hist.push_back(p);
    if (hist.size() > window_size_)
      hist.pop_front();

    double sx = 0.0, sy = 0.0;
    for (const auto &h : hist)
    {
      sx += h.x;
      sy += h.y;
    }
    double n = static_cast<double>(hist.size());
    return {sx / n, sy / n};
  }

  // 속도 히스토리 평균
  double avgSpeed(const std::deque<double> &hist) const
  {
    if (hist.empty())
      return 0.0;
    double s = 0.0;
    for (double v : hist)
      s += v;
    return s / static_cast<double>(hist.size());
  }

  // 이전 history 기반으로 raw c1, c2 를 A/B 에 매칭
  void associateClusters(const Point2D &c1,
                         const Point2D &c2,
                         Point2D &outA,
                         Point2D &outB)
  {
    // history 가 없다면 그냥 고정 순서로 세팅
    if (hist_A_.empty() || hist_B_.empty())
    {
      outA = c1;
      outB = c2;
      return;
    }

    // history 가 충분히 쌓이기 전: "가까운 쪽" 기준으로 매칭
    if (hist_A_.size() < 5 || hist_B_.size() < 5)
    {
      Point2D lastA = hist_A_.back();
      Point2D lastB = hist_B_.back();

      double cost1 = dist(c1, lastA) + dist(c2, lastB);
      double cost2 = dist(c2, lastA) + dist(c1, lastB);

      if (cost1 <= cost2)
      {
        outA = c1;
        outB = c2;
      }
      else
      {
        outA = c2;
        outB = c1;
      }
      return;
    }

    // 그 외: 이전 프레임 속도를 이용한 예측 기반 매칭
    Point2D prevA = hist_A_[hist_A_.size() - 2];
    Point2D lastA = hist_A_.back();
    Point2D prevB = hist_B_[hist_B_.size() - 2];
    Point2D lastB = hist_B_.back();

    Point2D vA_prev = sub(lastA, prevA);  // 이전 A 속도
    Point2D vB_prev = sub(lastB, prevB);  // 이전 B 속도

    // 이 속도로 한 프레임 더 갔다고 예측
    Point2D predA = add(lastA, vA_prev);
    Point2D predB = add(lastB, vB_prev);

    // case1: c1->A, c2->B
    double cost1 = dist(c1, predA) + dist(c2, predB);

    // case2: c1->B, c2->A
    double cost2 = dist(c2, predA) + dist(c1, predB);

    if (cost1 <= cost2)
    {
      outA = c1;
      outB = c2;
    }
    else
    {
      outA = c2;
      outB = c1;
    }
  }

  // ---------- 콜백 ----------

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
  {
    std::vector<Point2D> points;
    points.reserve(scan_msg->ranges.size());

    // 1) LaserScan → 점 집합 (로봇 기준 좌표)
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
    {
      double r = scan_msg->ranges[i];
      if (std::isnan(r) || std::isinf(r))
        continue;

      double angle = scan_msg->angle_min + static_cast<double>(i) * scan_msg->angle_increment;

      if (r < scan_range_min_ || r > scan_range_max_ ||
          angle < scan_angle_min_ || angle > scan_angle_max_)
      {
        continue;
      }

      Point2D p{r * std::cos(angle), r * std::sin(angle)};
      points.push_back(p);
    }

    if (points.size() < 2)
    {
      // 점이 너무 적으면 처리 안 함
      return;
    }

    // 2) K-means(k=2) → raw center 2개
    auto centers = kmeans2(points);
    Point2D c1 = centers.first;
    Point2D c2 = centers.second;

    // 3) 이전 history 기반으로 A/B identity 매칭 (raw)
    Point2D rawA, rawB;
    associateClusters(c1, c2, rawA, rawB);

    // 4) smoothing 후 최종 표시용 중심 계산
    Point2D smoothA = pushAndSmooth(hist_A_, rawA);
    Point2D smoothB = pushAndSmooth(hist_B_, rawB);

    // 5) "움직이는 클러스터 = 항상 A(빨강)" 이 되게 velocity 크기 비교
    if (hist_A_.size() >= 2 && hist_B_.size() >= 2)
    {
      Point2D lastA_prev = hist_A_[hist_A_.size() - 2];
      Point2D lastA      = hist_A_.back();
      Point2D lastB_prev = hist_B_[hist_B_.size() - 2];
      Point2D lastB      = hist_B_.back();

      Point2D vA_raw = sub(lastA, lastA_prev);
      Point2D vB_raw = sub(lastB, lastB_prev);

      double speedA = std::hypot(vA_raw.x, vA_raw.y);
      double speedB = std::hypot(vB_raw.x, vB_raw.y);

      // 최근 속도 히스토리 업데이트
      speed_hist_A_.push_back(speedA);
      if (speed_hist_A_.size() > speed_window_size_)
        speed_hist_A_.pop_front();

      speed_hist_B_.push_back(speedB);
      if (speed_hist_B_.size() > speed_window_size_)
        speed_hist_B_.pop_front();

      double avgA = avgSpeed(speed_hist_A_);
      double avgB = avgSpeed(speed_hist_B_);

      if (swap_cooldown_ > 0)
        --swap_cooldown_;

      // 평균 속도 차이가 margin보다 크고, 쿨다운이 0일 때만 swap
      if (avgB > avgA + speed_swap_margin_ && swap_cooldown_ == 0)
      {
        std::swap(hist_A_, hist_B_);
        std::swap(speed_hist_A_, speed_hist_B_);
        std::swap(smoothA, smoothB);
        swap_cooldown_ = swap_cooldown_max_;
      }
    }

    // 6) 두 중심 간 거리
    double dAB = dist(smoothA, smoothB);

    // 7) Marker 생성
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = scan_msg->header.frame_id;
    marker.header.stamp = scan_msg->header.stamp;
    marker.ns = "centers";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.15;
    marker.scale.y = 0.15;
    marker.color.a = 1.0;  // per-point color를 쓰더라도 alpha는 0이 아니어야 함

    geometry_msgs::msg::Point pA, pB;
    pA.x = smoothA.x;
    pA.y = smoothA.y;
    pA.z = 0.0;
    pB.x = smoothB.x;
    pB.y = smoothB.y;
    pB.z = 0.0;

    std::cout << std::fixed << std::setprecision(3);

    if (dAB < merge_threshold_)
    {
      // 너무 가까우면 하나로 merge (A 위치 기준, 색도 빨간색으로)
      marker.points.push_back(pA);

      std_msgs::msg::ColorRGBA c;
      c.r = 1.0f; c.g = 0.0f; c.b = 0.0f; c.a = 1.0f; // 빨간색
      marker.colors.push_back(c);

      std::cout << "Merged: ("
                << pA.x << ", " << pA.y
                << ")   dist=" << dAB << std::endl;
    }
    else
    {
      // A: 빨강 (항상 더 많이 움직이는 쪽)
      marker.points.push_back(pA);
      std_msgs::msg::ColorRGBA cA;
      cA.r = 1.0f; cA.g = 0.0f; cA.b = 0.0f; cA.a = 1.0f;
      marker.colors.push_back(cA);

      // B: 파랑 (더 정적인 쪽)
      marker.points.push_back(pB);
      std_msgs::msg::ColorRGBA cB;
      cB.r = 0.0f; cB.g = 0.0f; cB.b = 1.0f; cB.a = 1.0f;
      marker.colors.push_back(cB);

      std::cout << "A(" << pA.x << ", " << pA.y << ")   "
                << "B(" << pB.x << ", " << pB.y << ")   "
                << "dist=" << dAB << std::endl;
    }

    marker_pub_->publish(marker);
  }

  // ---------- 멤버 변수들 ----------

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  std::deque<Point2D> hist_A_;
  std::deque<Point2D> hist_B_;

  const std::size_t window_size_;
  const double merge_threshold_; // [m]

  // 속도 기반 히스테리시스
  std::deque<double> speed_hist_A_;
  std::deque<double> speed_hist_B_;
  const std::size_t speed_window_size_;
  const double speed_swap_margin_;
  int swap_cooldown_;
  const int swap_cooldown_max_;

  double scan_range_min_;
  double scan_range_max_;
  double scan_angle_min_;
  double scan_angle_max_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
