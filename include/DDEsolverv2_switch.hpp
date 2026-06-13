#pragma once

#include "radix_heap.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <type_traits>
#include <vector>

namespace DDE {

// -------------------------------------------------------------------------
// Zero-Allocation Taylor Polynomial (Truncated to compile-time Degree K)
// -------------------------------------------------------------------------
template <typename T, size_t K> struct TaylorPoly {
  std::array<T, K> c{};

  TaylorPoly() = default;
  explicit TaylorPoly(T val) noexcept { c[0] = val; }

  [[nodiscard]] T nom() const noexcept { return c[0]; }

  TaylorPoly &operator+=(const TaylorPoly &rhs) noexcept {
#pragma GCC unroll 8
    for (size_t i = 0; i < K; ++i)
      c[i] += rhs.c[i];
    return *this;
  }
  TaylorPoly &operator-=(const TaylorPoly &rhs) noexcept {
#pragma GCC unroll 8
    for (size_t i = 0; i < K; ++i)
      c[i] -= rhs.c[i];
    return *this;
  }
  TaylorPoly &operator+=(T s) noexcept {
    c[0] += s;
    return *this;
  }
  TaylorPoly &operator-=(T s) noexcept {
    c[0] -= s;
    return *this;
  }
  TaylorPoly &operator*=(T s) noexcept {
#pragma GCC unroll 8
    for (size_t i = 0; i < K; ++i)
      c[i] *= s;
    return *this;
  }
  TaylorPoly &operator/=(T s) noexcept {
    T inv = T(1) / s;
#pragma GCC unroll 8
    for (size_t i = 0; i < K; ++i)
      c[i] *= inv;
    return *this;
  }

  [[nodiscard]] inline TaylorPoly
  operator+(const TaylorPoly &rhs) const noexcept {
    TaylorPoly res(*this);
    res += rhs;
    return res;
  }
  [[nodiscard]] inline TaylorPoly
  operator-(const TaylorPoly &rhs) const noexcept {
    TaylorPoly res(*this);
    res -= rhs;
    return res;
  }
  [[nodiscard]] inline TaylorPoly
  operator*(const TaylorPoly &rhs) const noexcept {
    TaylorPoly res;
    for (size_t i = 0; i < K; ++i) {
#pragma GCC unroll 8
      for (size_t j = 0; j < K - i; ++j)
        res.c[i + j] += c[i] * rhs.c[j];
    }
    return res;
  }
  [[nodiscard]] inline TaylorPoly
  operator/(const TaylorPoly &rhs) const noexcept {
    TaylorPoly res;
    T inv0 = T(1) / rhs.c[0];
    for (size_t k = 0; k < K; ++k) {
      T sum = c[k];
      for (size_t j = 1; j <= k; ++j)
        sum -= rhs.c[j] * res.c[k - j];
      res.c[k] = sum * inv0;
    }
    return res;
  }
  [[nodiscard]] inline TaylorPoly operator-() const noexcept {
    TaylorPoly res(*this);
    res *= T(-1);
    return res;
  }
};

template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K>
operator+(T s, const TaylorPoly<T, K> &b) noexcept {
  TaylorPoly<T, K> res(b);
  res.c[0] += s;
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K> operator+(const TaylorPoly<T, K> &a,
                                                T s) noexcept {
  TaylorPoly<T, K> res(a);
  res.c[0] += s;
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K>
operator-(T s, const TaylorPoly<T, K> &b) noexcept {
  TaylorPoly<T, K> res;
  res.c[0] = s;
#pragma GCC unroll 8
  for (size_t i = 0; i < K; ++i)
    res.c[i] -= b.c[i];
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K> operator-(const TaylorPoly<T, K> &a,
                                                T s) noexcept {
  TaylorPoly<T, K> res(a);
  res.c[0] -= s;
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K>
operator*(T s, const TaylorPoly<T, K> &b) noexcept {
  TaylorPoly<T, K> res(b);
  res *= s;
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K> operator*(const TaylorPoly<T, K> &a,
                                                T s) noexcept {
  TaylorPoly<T, K> res(a);
  res *= s;
  return res;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K>
operator/(T s, const TaylorPoly<T, K> &b) noexcept {
  return TaylorPoly<T, K>(s) / b;
}
template <typename T, size_t K>
[[nodiscard]] inline TaylorPoly<T, K> operator/(const TaylorPoly<T, K> &a,
                                                T s) noexcept {
  TaylorPoly<T, K> res(a);
  res /= s;
  return res;
}

// -------------------------------------------------------------------------
// Coordinate-Wise Asynchronous DDE Solver
// -------------------------------------------------------------------------
template <size_t NumVars, size_t Degree, typename FunctorF, typename FunctorTau,
          typename FunctorHistory, bool UseRadixHeap = false>
class AsynchronousTaylorDDE {
public:
  static constexpr size_t PolySize = Degree + 1;
  using Poly = TaylorPoly<double, PolySize>;
  using StateArray = std::array<double, NumVars>;
  using DependencyGraph = std::array<std::vector<int>, NumVars>;

  AsynchronousTaylorDDE(FunctorF f, FunctorTau tau, FunctorHistory history,
                        StateArray x0, DependencyGraph deps, double t0,
                        double epsilon, double tau_max = 0.15)
      : f_func(std::move(f)), tau_func(std::move(tau)),
        history_func(std::move(history)), x0(x0), dependents(std::move(deps)),
        t0(t0), epsilon(epsilon), tau_max(tau_max), event_queue(NumVars) {
    for (auto &rb : history_segments) {
      rb = RingBuffer<Segment>(65536);
    }
  }

  std::tuple<std::vector<double>, std::vector<StateArray>,
             std::array<uint64_t, NumVars>>
  solve(double T_max, size_t num_report_points = 1000) {
    std::vector<double> times;
    std::vector<StateArray> states;
    std::array<uint64_t, NumVars> event_counts;
    event_counts.fill(0);

    times.reserve(num_report_points + 1);
    states.reserve(num_report_points + 1);

    double current_time = t0;
    double dt_report = (T_max - t0) / static_cast<double>(num_report_points);
    double next_report_time = t0;

    // Initial setup
    for (size_t i = 0; i < NumVars; ++i) {
      auto coeffs = _compute_single_ad(i, current_time, x0[i]);
      double dt = _step_size(coeffs[Degree]);
      history_segments[i].push_back({current_time, current_time + dt, coeffs});
      event_queue.update_or_push(current_time + dt, static_cast<int>(i));
      event_counts[i]++;
    }

    while (!event_queue.empty() && event_queue.top().time <= T_max) {
      const auto ev = event_queue.top();

      while (next_report_time <= ev.time && next_report_time <= T_max) {
        times.push_back(next_report_time);
        states.push_back(_get_current_state(next_report_time));
        next_report_time += dt_report;
      }

      event_queue.pop();

      current_time = ev.time;
      const int v = ev.var_idx;

      auto &last_seg = history_segments[v].back();
      const double new_val =
          _eval_poly(last_seg.coeffs, current_time, last_seg.start_t);
      last_seg.end_t = current_time;

      auto coeffs = _compute_single_ad(v, current_time, new_val);
      double dt = _step_size(coeffs[Degree]);
      history_segments[v].push_back({current_time, current_time + dt, coeffs});

      event_queue.update_or_push(current_time + dt, v);
      event_counts[v]++;

      for (int j : dependents[v]) {
        if (j == v)
          continue;

        const auto &seg_j = history_segments[j].back();
        if (seg_j.end_t <= current_time)
          continue;

        const double poly_deriv =
            _eval_poly_deriv(seg_j.coeffs, current_time, seg_j.start_t);

        auto get_x_nom = [&](size_t idx) -> Poly {
          if (idx == static_cast<size_t>(v))
            return Poly(new_val);
          if (history_segments[idx].empty())
            return Poly(x0[idx]);
          const auto &seg_idx = history_segments[idx].back();
          return Poly(
              _eval_poly(seg_idx.coeffs, current_time, seg_idx.start_t));
        };

        auto get_x_delay_nom = [&](size_t idx) -> Poly {
          Poly t_ad(current_time);
          Poly tau_ad = tau_func(idx, t_ad, get_x_nom);
          double t_d_nom = current_time - tau_ad.c[0];
          const Segment *seg = _find_segment(idx, t_d_nom);
          if (seg)
            return Poly(_eval_poly(seg->coeffs, t_d_nom, seg->start_t));
          else
            return history_func(idx, Poly(t_d_nom));
        };

        Poly f_ad_j = f_func(j, get_x_nom, get_x_delay_nom);
        double true_deriv = f_ad_j.c[0];

        if (std::abs(true_deriv - poly_deriv) * (seg_j.end_t - current_time) >
            epsilon) {
          event_queue.update_or_push(current_time, j);
        }
      }
    }

    while (next_report_time <= T_max) {
      times.push_back(next_report_time);
      states.push_back(_get_current_state(next_report_time));
      next_report_time += dt_report;
    }

    return {times, states, event_counts};
  }

private:
  static constexpr double kInvPolySize = 1.0 / static_cast<double>(PolySize);

  FunctorF f_func;
  FunctorTau tau_func;
  FunctorHistory history_func;
  StateArray x0;
  DependencyGraph dependents;
  double t0;
  double epsilon;
  double tau_max;

  struct Segment {
    double start_t, end_t;
    std::array<double, PolySize> coeffs;
  };

  template <typename T> class RingBuffer {
    std::vector<T> buffer;
    size_t head = 0, tail = 0, count = 0;

  public:
    RingBuffer() = default;
    explicit RingBuffer(size_t capacity) : buffer(capacity) {}

    void push_back(const T &item) {
      if (count == buffer.size()) {
        buffer[tail] = item;
        head = (head + 1) % buffer.size();
        tail = (tail + 1) % buffer.size();
      } else {
        buffer[tail] = item;
        tail = (tail + 1) % buffer.size();
        count++;
      }
    }

    void pop_front() {
      if (count > 0) {
        head = (head + 1) % buffer.size();
        count--;
      }
    }

    T &back() { return buffer[(tail == 0 ? buffer.size() : tail) - 1]; }
    const T &back() const {
      return buffer[(tail == 0 ? buffer.size() : tail) - 1];
    }
    const T &operator[](size_t i) const {
      return buffer[(head + i) % buffer.size()];
    }

    size_t size() const { return count; }
    bool empty() const { return count == 0; }
  };

  std::array<RingBuffer<Segment>, NumVars> history_segments;
  mutable std::array<size_t, NumVars> history_cursors{0};

  struct SimpleEvent {
    double time;
    int var_idx;
  };

  // =========================================================================
  // QUEUE 1: ZERO-ALLOCATION UPDATABLE MIN-HEAP (O(log N), O(N) Space)
  // =========================================================================
  class UpdatableMinHeap {
    std::vector<SimpleEvent> heap;
    std::vector<int> pos;

    void sift_up(int i) {
      while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[i].time < heap[p].time) {
          std::swap(heap[i], heap[p]);
          pos[heap[i].var_idx] = i;
          pos[heap[p].var_idx] = p;
          i = p;
        } else
          break;
      }
    }

    void sift_down(int i) {
      int n = heap.size();
      while (2 * i + 1 < n) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = left;
        if (right < n && heap[right].time < heap[left].time)
          smallest = right;

        if (heap[smallest].time < heap[i].time) {
          std::swap(heap[i], heap[smallest]);
          pos[heap[i].var_idx] = i;
          pos[heap[smallest].var_idx] = smallest;
          i = smallest;
        } else
          break;
      }
    }

  public:
    explicit UpdatableMinHeap(size_t capacity) {
      heap.reserve(capacity);
      pos.assign(capacity, -1);
    }

    void update_or_push(double time, int var_idx) {
      if (pos[var_idx] == -1) {
        int i = heap.size();
        heap.push_back({time, var_idx});
        pos[var_idx] = i;
        sift_up(i);
      } else {
        int i = pos[var_idx];
        double old_time = heap[i].time;
        heap[i].time = time;
        if (time < old_time)
          sift_up(i);
        else
          sift_down(i);
      }
    }

    [[nodiscard]] SimpleEvent top() const noexcept { return heap.front(); }

    void pop() {
      int n = heap.size();
      pos[heap[0].var_idx] = -1;
      if (n > 1) {
        heap[0] = heap[n - 1];
        pos[heap[0].var_idx] = 0;
        heap.pop_back();
        sift_down(0);
      } else {
        heap.pop_back();
      }
    }

    [[nodiscard]] bool empty() const noexcept { return heap.empty(); }
  };

  // =========================================================================
  // QUEUE 2: RADIX HEAP WRAPPER (Amortized O(1), Lazy Deletion)
  // =========================================================================
  class RadixHeapWrapper {
    struct EventPayload {
      uint64_t id;
      int var_idx;
    };
    radix_heap::pair_radix_heap<double, EventPayload> rq;
    std::vector<int64_t> valid_ids;
    uint64_t event_counter = 0;

    void clean_lazy() {
      while (!rq.empty()) {
        auto payload = rq.top_value();
        if (valid_ids[payload.var_idx] == static_cast<int64_t>(payload.id)) {
          break; // Found the valid, most recent event for this variable
        }
        rq.pop(); // Phantom event, discard
      }
    }

  public:
    explicit RadixHeapWrapper(size_t capacity) {
      valid_ids.assign(capacity, -1);
    }

    void update_or_push(double time, int var_idx) {
      uint64_t id = ++event_counter;
      valid_ids[var_idx] = static_cast<int64_t>(id);
      rq.push(time, {id, var_idx});
    }

    [[nodiscard]] SimpleEvent top() {
      clean_lazy();
      return {rq.top_key(), rq.top_value().var_idx};
    }

    void pop() { rq.pop(); }

    [[nodiscard]] bool empty() {
      clean_lazy();
      return rq.empty();
    }
  };

  // =========================================================================
  // COMPILE-TIME QUEUE SELECTION
  // =========================================================================
  using EventQueueType =
      std::conditional_t<UseRadixHeap, RadixHeapWrapper, UpdatableMinHeap>;
  EventQueueType event_queue;

  // ---- Binomial Shift Mechanics (O(1) Time Warping) -----------------------
  static constexpr double nCr(int n, int k) {
    if (k < 0 || k > n)
      return 0;
    if (k == 0 || k == n)
      return 1;
    double res = 1;
    for (int i = 1; i <= k; ++i)
      res = res * (n - i + 1) / i;
    return res;
  }

  [[nodiscard]] inline Poly _shift_poly(const std::array<double, PolySize> &c,
                                        double dt) const noexcept {
    Poly res;
    for (int k = 0; k < PolySize; ++k) {
      double term = c[k];
      double dt_pow = 1.0;
      for (int m = 0; m <= k; ++m) {
        res.c[k - m] += term * nCr(k, m) * dt_pow;
        dt_pow *= dt;
      }
    }
    return res;
  }

  // ---- Polynomial helpers -------------------------------------------------
  [[nodiscard]] inline double _eval_poly(const std::array<double, PolySize> &c,
                                         double t,
                                         double base_t) const noexcept {
    double dt = t - base_t, val = 0.0;
    for (int k = static_cast<int>(PolySize) - 1; k >= 0; --k)
      val = val * dt + c[k];
    return val;
  }

  [[nodiscard]] inline double
  _eval_poly_deriv(const std::array<double, PolySize> &c, double t,
                   double base_t) const noexcept {
    double dt = t - base_t, val = 0.0;
    for (int k = static_cast<int>(PolySize) - 1; k >= 1; --k)
      val = val * dt + static_cast<double>(k) * c[k];
    return val;
  }

  [[nodiscard]] inline double _step_size(double leading_coeff) const noexcept {
    double M = std::max(1e-10, std::abs(leading_coeff));
    return std::max(std::exp(std::log(epsilon / M) * kInvPolySize), 1e-12);
  }

  [[nodiscard]] const Segment *_find_segment(size_t var,
                                             double t_d) const noexcept {
    const auto &segs = history_segments[var];
    if (segs.empty())
      return nullptr;
    size_t &idx = history_cursors[var];
    if (idx >= segs.size())
      idx = segs.size() - 1;
    if (t_d >= segs[idx].start_t && t_d <= segs[idx].end_t + 1e-10)
      return &segs[idx];
    while (idx + 1 < segs.size() && t_d > segs[idx].end_t + 1e-10) {
      idx++;
      if (t_d >= segs[idx].start_t && t_d <= segs[idx].end_t + 1e-10)
        return &segs[idx];
    }
    while (idx > 0 && t_d < segs[idx].start_t) {
      idx--;
      if (t_d >= segs[idx].start_t && t_d <= segs[idx].end_t + 1e-10)
        return &segs[idx];
    }
    return nullptr;
  }

  [[nodiscard]] StateArray _get_current_state(double t) const {
    StateArray state;
    for (size_t i = 0; i < NumVars; ++i) {
      if (!history_segments[i].empty()) {
        const auto &seg = history_segments[i].back();
        state[i] = _eval_poly(seg.coeffs, t, seg.start_t);
      } else {
        state[i] = x0[i];
      }
    }
    return state;
  }

  // ---- Coordinate-Wise Automatic Differentiation --------------------------
  std::array<double, PolySize> _compute_single_ad(size_t v, double current_time,
                                                  double val_v) {
    Poly x_ad_v;
    x_ad_v.c[0] = val_v;
    Poly t_ad;
    t_ad.c[0] = current_time;
    t_ad.c[1] = 1.0;

    auto get_x = [&](size_t j) -> Poly {
      if (j == v)
        return x_ad_v;
      if (history_segments[j].empty())
        return Poly(x0[j]);
      const auto &seg = history_segments[j].back();
      return _shift_poly(seg.coeffs, current_time - seg.start_t);
    };

    auto get_x_delay = [&](size_t j) -> Poly {
      Poly tau_ad = tau_func(j, t_ad, get_x);
      Poly t_delay_ad = t_ad - tau_ad;
      double t_d_nom = t_delay_ad.c[0];
      const Segment *seg = _find_segment(j, t_d_nom);

      if (seg) {
        Poly dt_ad = t_delay_ad - seg->start_t;
        Poly val_ad(seg->coeffs.back());
#pragma GCC unroll 8
        for (int p = static_cast<int>(PolySize) - 2; p >= 0; --p)
          val_ad = val_ad * dt_ad + seg->coeffs[p];
        return val_ad;
      } else {
        return history_func(j, t_delay_ad);
      }
    };

    for (size_t k = 0; k < Degree; ++k) {
      Poly f_ad = f_func(v, get_x, get_x_delay);
      x_ad_v.c[k + 1] = f_ad.c[k] / static_cast<double>(k + 1);
    }
    return x_ad_v.c;
  }
};

} // namespace DDE
