#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include "DDEsolverv2_switch.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <size_t NumVars>
void export_solver_advantages(
    const std::string &prefix, const std::vector<double> &times,
    const std::vector<std::array<double, NumVars>> &states,
    const std::array<uint64_t, NumVars> &event_counts) // <-- Accept the counts
{
  std::cout << " -> Saving diagnostic logs for [" << prefix << "]...\n";

  // 1. Save Trajectories
  std::ofstream state_file(prefix + "_trajectories.csv");
  state_file << "time";
  size_t vars_to_log = std::min(NumVars, size_t(20));
  for (size_t j = 0; j < vars_to_log; ++j)
    state_file << ",x" << j;
  state_file << "\n" << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < times.size(); ++i) {
    state_file << times[i];
    for (size_t j = 0; j < vars_to_log; ++j)
      state_file << "," << states[i][j];
    state_file << "\n";
  }
  state_file.close();

  // 2. Save Event Distribution (Multi-Rate Proof!)
  std::ofstream dist_file(prefix + "_event_distribution.csv");
  dist_file << "var_idx,total_events\n";
  for (size_t j = 0; j < NumVars; ++j) {
    dist_file << j << "," << event_counts[j] << "\n";
  }
  dist_file.close();

  std::cout << "    * Saved " << prefix << "_trajectories.csv\n";
  std::cout << "    * Saved " << prefix << "_event_distribution.csv\n";
}

// ============================================================================
// 1. BENCHMARK DEFINITION: Advection-Reaction DPDE
// ============================================================================
constexpr size_t NumVars_DPDE = 10000;
constexpr double RHO_COEFF = 1.0;
constexpr double TAU_DPDE_VAL = 0.5;
constexpr double L_DOM = 1.0;
constexpr double DX_INV = static_cast<double>(NumVars_DPDE + 1);
constexpr double VELOCITY = 1.0;

// Inside benchmarks.cpp
auto f_dpde = [](size_t i, const auto &x, const auto &x_delay) {
  using T = typename std::decay_t<decltype(x(0))>;

  size_t left_idx = (i == 0) ? NumVars_DPDE - 1 : i - 1;

  auto dxdz = (x(i) - x(left_idx)) * DX_INV;
  T reaction = x(i) * (1.0 - x_delay(i)) * RHO_COEFF;
  return -VELOCITY * dxdz + reaction;
};
auto tau_dpde = [](size_t i, const auto &t, const auto &x) {
  return decltype(t)(TAU_DPDE_VAL);
};
auto history_dpde = [](size_t i, const auto &t) {
  // Use decay_t to strip 'const' and '&' from the deduced type
  using T = typename std::decay_t<decltype(t)>;
  double x_coord = (static_cast<double>(i) + 1.0) / DX_INV;
  T s = T(std::sin(M_PI * x_coord / L_DOM));
  return s * s;
};

// ============================================================================
// 2. BENCHMARK DEFINITION: Sparse Multi-Rate Network
// ============================================================================
constexpr size_t NumVars_MR = 10000;
constexpr double TAU_MR_VAL = 0.5;

auto f_mr = [](size_t i, const auto &x, const auto &x_delay) {
  double alpha = (i < 100) ? 50.0 : 0.01;
  auto decay = x(i) * (-alpha);
  auto coupling = decltype(decay)(0.0);

  size_t remainder = i % 100;
  size_t first_j = (remainder == 0) ? 0 : (100 - remainder);
  for (size_t j = first_j; j < NumVars_MR; j += 100) {
    double weight = 0.05 * std::cos(static_cast<double>(i + j));
    auto x_del = x_delay(j);
    auto activation = x_del / (1.0 + x_del * x_del);
    coupling += activation * weight;
  }
  return decay + coupling;
};
auto tau_mr = [](size_t i, const auto &t, const auto &x) {
  return decltype(t)(TAU_MR_VAL);
};
auto history_mr = [](size_t i, const auto &t) {
  using T = typename std::decay_t<decltype(t)>;
  return T(0.1);
};

// ============================================================================
// 3. BENCHMARK DEFINITION: State-Dependent Ring Network
// ============================================================================
constexpr size_t NumVars_SD = 10000;
constexpr double BETA_COEFF = 5.0;
constexpr double GAMMA_COEFF = 1.0;

auto f_sd = [](size_t i, const auto &x, const auto &x_delay) {
  size_t prev_idx = (i == 0) ? NumVars_SD - 1 : i - 1;
  auto x_delayed_neighbor = x_delay(prev_idx);
  auto hill_denom =
      1.0 + (x_delayed_neighbor * x_delayed_neighbor * x_delayed_neighbor);
  auto production = BETA_COEFF / hill_denom;
  auto decay = x(i) * (-GAMMA_COEFF);
  return production + decay;
};
auto tau_sd = [](size_t i, const auto &t, const auto &x) {
  // The delay is now locally state-dependent on the variable itself
  auto x_local = x(i);
  return 1.0 + (x_local * 0.5) / (1.0 + x_local * x_local);
};
auto history_sd = [](size_t i, const auto &t) {
  using T = typename std::decay_t<decltype(t)>;
  return T(0.5);
};

// ============================================================================
// 4. BENCHMARK DEFINITION: Propagating Pulse Chain (Dynamic Multi-Rate)
// ============================================================================
constexpr size_t NumVars_Pulse = 10000;
constexpr double TAU_PULSE_VAL = 0.5;

auto f_pulse = [](size_t i, const auto &x, const auto &x_delay) {
  using T = typename std::decay_t<decltype(x(0))>;

  T decay = x(i) * (-1.0); // Natural decay

  if (i == 0) {
    return decay; // Node 0 receives no input, just decays from its initial
                  // state
  } else {
    auto x_prev = x_delay(i - 1);
    // A non-linear Hill activation function (Degree 2)
    auto activation = (2.0 * x_prev * x_prev) / (0.25 + x_prev * x_prev);
    return decay + activation;
  }
};

auto tau_pulse = [](size_t i, const auto &t, const auto &x) {
  return decltype(t)(TAU_PULSE_VAL);
};

auto history_pulse = [](size_t i, const auto &t) {
  using T = typename std::decay_t<decltype(t)>;
  return (i == 0) ? T(1.0) : T(0.0); // Only the first node starts excited
};

// ============================================================================
// 5. BENCHMARK DEFINITION: Diffusion DPDE (Parabolic Localized Energy)
// ============================================================================
// Change this specific constant for the Diffusion block
constexpr size_t NumVars_Diff = 50; // Scaled to 100 to mitigate OOM
constexpr double ALPHA_DIFF = 1.2;
constexpr double BETA_DIFF = 0.4;
constexpr double TAU_DIFF_VAL = 0.15;
constexpr double DX_INV_DIFF = NumVars_Diff + 1.0; // Equals 101.0

auto f_diff = [](size_t i, const auto &x, const auto &x_delay) {
  using T = typename std::decay_t<decltype(x(0))>;

  size_t left_idx = (i == 0) ? NumVars_Diff - 1 : i - 1;
  size_t right_idx = (i == NumVars_Diff - 1) ? 0 : i + 1;

  // Stencil multiplied by (DX_INV * DX_INV) to incorporate physical stiffness
  // safely
  auto diffusion_stencil = ALPHA_DIFF *
                           (x(right_idx) - 2.0 * x(i) + x(left_idx)) *
                           (DX_INV_DIFF * DX_INV_DIFF);
  T delayed_loss = BETA_DIFF * x(i) * x_delay(i);

  return diffusion_stencil - delayed_loss;
};

auto tau_diff = [](size_t i, const auto &t, const auto &x) {
  return decltype(t)(TAU_DIFF_VAL);
};

auto history_diff = [](size_t i, const auto &t) {
  using T = typename std::decay_t<decltype(t)>;
  return T(0.0); // System at rest for t < 0
};

// ============================================================================
// CORE EXECUTION PIPELINE
// ============================================================================
int main() {
  std::cout << "=======================================================\n";
  std::cout << "  RUNNING FULLY ASYNCHRONOUS COORDINATE-WISE SOLVER\n";
  std::cout << "=======================================================\n\n";

  double epsilon = 1e-10;
  double T_max = 5.0;

  // --- Benchmark 1: Advection-Reaction DPDE (N=10000, Degree=6) ---
  {
    constexpr size_t Degree_DPDE = 4;
    auto x0_dpde = std::make_unique<std::array<double, NumVars_DPDE>>();
    for (size_t i = 0; i < NumVars_DPDE; ++i) {
      double x_coord = (static_cast<double>(i) + 1.0) / DX_INV;
      double s = std::sin(M_PI * x_coord / L_DOM);
      (*x0_dpde)[i] = s * s; // SQUARE IT!
    }

    std::array<std::vector<int>, NumVars_DPDE> deps_dpde;
    for (size_t i = 0; i < NumVars_DPDE; ++i) {
      deps_dpde[i].push_back(i);
      size_t right_idx = (i == NumVars_DPDE - 1) ? 0 : i + 1;
      deps_dpde[i].push_back(right_idx); // Changing left boundary alters right
                                         // node's gradient instantly
    }

    using SolverType1 =
        DDE::AsynchronousTaylorDDE<NumVars_DPDE, Degree_DPDE, decltype(f_dpde),
                                   decltype(tau_dpde), decltype(history_dpde),
                                   true>;
    auto solver1 =
        std::make_unique<SolverType1>(f_dpde, tau_dpde, history_dpde, *x0_dpde,
                                      deps_dpde, 0.0, epsilon, TAU_DPDE_VAL);

    std::cout << "Executing Benchmark 1 [Advection DPDE] (N=" << NumVars_DPDE
              << ", Degree=" << Degree_DPDE << ")..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto res = solver1->solve(T_max);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Done. (Execution Time: " << elapsed.count() << " seconds)\n";
    export_solver_advantages<NumVars_DPDE>("dpde", std::get<0>(res),
                                           std::get<1>(res), std::get<2>(res));
  }

  // --- Benchmark 2: Multi-Rate Network (N=10000, Degree=4) ---
  {
    constexpr size_t Degree_MR = 4;
    auto x0_mr = std::make_unique<std::array<double, NumVars_MR>>();
    x0_mr->fill(0.1);

    std::array<std::vector<int>, NumVars_MR> deps_mr;
    for (size_t i = 0; i < NumVars_MR; ++i) {
      deps_mr[i].push_back(i); // Interactions are purely delayed, so no
                               // instantaneous triggers are required!
    }

    using SolverType2 =
        DDE::AsynchronousTaylorDDE<NumVars_MR, Degree_MR, decltype(f_mr),
                                   decltype(tau_mr), decltype(history_mr),
                                   true>;
    auto solver2 = std::make_unique<SolverType2>(
        f_mr, tau_mr, history_mr, *x0_mr, deps_mr, 0.0, epsilon, TAU_MR_VAL);

    std::cout << "\nExecuting Benchmark 2 [Multi-Rate] (N=" << NumVars_MR
              << ", Degree=" << Degree_MR << ")..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto res = solver2->solve(T_max);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Done. (Execution Time: " << elapsed.count() << " seconds)\n";
    export_solver_advantages<NumVars_MR>("multirate", std::get<0>(res),
                                         std::get<1>(res), std::get<2>(res));
  }

  // --- Benchmark 3: State-Dependent Delay (N=10000, Degree=4) ---
  {
    constexpr size_t Degree_SD = 4;
    auto x0_sd = std::make_unique<std::array<double, NumVars_SD>>();
    x0_sd->fill(0.5);

    std::array<std::vector<int>, NumVars_SD> deps_sd;
    for (size_t i = 0; i < NumVars_SD; ++i) {
      deps_sd[i].push_back(i);
    }

    using SolverType3 =
        DDE::AsynchronousTaylorDDE<NumVars_SD, Degree_SD, decltype(f_sd),
                                   decltype(tau_sd), decltype(history_sd),
                                   true>;
    auto solver3 = std::make_unique<SolverType3>(
        f_sd, tau_sd, history_sd, *x0_sd, deps_sd, 0.0, epsilon, 1.5);

    std::cout << "\nExecuting Benchmark 3 [State-Dependent] (N=" << NumVars_SD
              << ", Degree=" << Degree_SD << ")..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto res = solver3->solve(T_max);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Done. (Execution Time: " << elapsed.count() << " seconds)\n";
    export_solver_advantages<NumVars_SD>("statedep", std::get<0>(res),
                                         std::get<1>(res), std::get<2>(res));
  }

  // --- Benchmark 4: Propagating Pulse Chain (N=10000, Degree=4) ---
  {
    constexpr size_t Degree_Pulse = 4;

    auto x0_pulse = std::make_unique<std::array<double, NumVars_Pulse>>();
    x0_pulse->fill(0.0);
    (*x0_pulse)[0] = 1.0; // Trigger the domino effect at Node 0

    std::array<std::vector<int>, NumVars_Pulse> deps_pulse;
    for (size_t i = 0; i < NumVars_Pulse; ++i) {
      deps_pulse[i].push_back(i);
      if (i + 1 < NumVars_Pulse) {
        // Structural link: Node i's delayed state drives Node i+1.
        // This guarantees Node i+1 is woken up when the wavefront arrives!
        deps_pulse[i].push_back(i + 1);
      }
    }

    using SolverType3 =
        DDE::AsynchronousTaylorDDE<NumVars_Pulse, Degree_Pulse,
                                   decltype(f_pulse), decltype(tau_pulse),
                                   decltype(history_pulse), true>;
    auto solver3 =
        std::make_unique<SolverType3>(f_pulse, tau_pulse, history_pulse,
                                      *x0_pulse, deps_pulse, 0.0, epsilon, 1.5);

    std::cout << "\nExecuting Benchmark 4 [Propagating Pulse] (N="
              << NumVars_Pulse << ", Degree=" << Degree_Pulse << ")..."
              << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto res = solver3->solve(T_max, 1000); // 1000 reporting points
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Done. (Execution Time: " << elapsed.count() << " seconds)\n";
    export_solver_advantages<NumVars_Pulse>("prop_pulse", std::get<0>(res),
                                            std::get<1>(res), std::get<2>(res));
  }

  // --- Benchmark 5: Diffusion DPDE (N=10000, Degree=4) ---
  {
    constexpr size_t Degree_Diff = 4;

    // 1. Initialize State with a sharp localized Gaussian pulse
    auto x0_diff = std::make_unique<std::array<double, NumVars_Diff>>();
    size_t center = NumVars_Diff / 2;
    double width = 25.0;
    for (size_t i = 0; i < NumVars_Diff; ++i) {
      double dist = static_cast<double>(i) - static_cast<double>(center);
      (*x0_diff)[i] = std::exp(-(dist * dist) / (2.0 * width * width));
    }

    // 2. Build 1D Structural Dependency Graph
    // In a diffusion stencil, modifying node i instantly affects its left and
    // right neighbors
    std::array<std::vector<int>, NumVars_Diff> deps_diff;
    for (size_t i = 0; i < NumVars_Diff; ++i) {
      deps_diff[i].push_back(i);
      size_t left_idx = (i == 0) ? NumVars_Diff - 1 : i - 1;
      size_t right_idx = (i == NumVars_Diff - 1) ? 0 : i + 1;
      deps_diff[i].push_back(
          left_idx); // Instantaneous gradient update to the left
      deps_diff[i].push_back(
          right_idx); // Instantaneous gradient update to the right
    }

    using SolverType5 =
        DDE::AsynchronousTaylorDDE<NumVars_Diff, Degree_Diff, decltype(f_diff),
                                   decltype(tau_diff), decltype(history_diff),
                                   false>;
    auto solver5 = std::make_unique<SolverType5>(
        f_diff, tau_diff, history_diff, *x0_diff, deps_diff, 0.0, epsilon, 1.5);

    std::cout << "\nExecuting Benchmark 5 [Diffusion DPDE] (N=" << NumVars_Diff
              << ", Degree=" << Degree_Diff << ")..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto res = solver5->solve(T_max);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Done. (Execution Time: " << elapsed.count() << " seconds)\n";
    export_solver_advantages<NumVars_Diff>("diffusion", std::get<0>(res),
                                           std::get<1>(res), std::get<2>(res));
  }

  std::cout << "\nAll SISC validation configurations executed successfully.\n";
  return 0;
}
