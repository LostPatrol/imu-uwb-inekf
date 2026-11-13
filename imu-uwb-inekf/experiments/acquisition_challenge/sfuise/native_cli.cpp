// Standalone causal I/O for the pinned SFUISE author numerical implementation.
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include "AuthorFusion.h"

// One ROS-equivalent fusion poll occurs every millisecond.
constexpr int64_t POLL_NS = 1000000;
// The author config uses ten control points per second.
constexpr int64_t KNOT_NS = 100000000;

struct Packet {
    double t;
    bool imu;
    std::vector<double> values;
};

// Split the simple numeric CSV schema used by the common experiment streams.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) result.push_back(field);
    return result;
}

// Read numeric CSV rows into named fields without reordering measurements.
std::vector<std::map<std::string, double>> readCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open " + path);
    std::string line;
    std::getline(input, line);
    auto header = split(line);
    if (!header.empty() && header.back().back() == '\r') header.back().pop_back();
    std::vector<std::map<std::string, double>> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto fields = split(line);
        std::map<std::string, double> row;
        for (size_t i = 0; i < header.size() && i < fields.size(); ++i)
            row[header[i]] = std::stod(fields[i]);
        result.push_back(row);
    }
    return result;
}

// Apply the author's Walk1 numerical settings; range corrections remain zero.
void configure(SplineFusion& fusion, const std::map<std::string, std::string>& args) {
    fusion.if_anchor_ini = true;
    fusion.average_runtime = 0;
    fusion.window_count = 0;
    fusion.solver_flag = SplineFusion::INITIAL;
    fusion.if_tdoa = false;
    fusion.if_uwb_only = false;
    fusion.dt_ns = KNOT_NS;
    fusion.bag_start_time = 0;
    fusion.n_window_calib = 50;
    fusion.window_size = 100;
    fusion.max_iter = 5;
    auto& p = fusion.param;
    p.if_opt_g = true;
    p.if_opt_transform = true;
    p.w_uwb = 7;
    p.w_acc = p.w_gyro = p.w_bias_acc = p.w_bias_gyro = 1;
    p.accel_var_inv.setConstant(7.8483);
    p.gyro_var_inv.setConstant(27.5646);
    p.bias_accel_var_inv.setConstant(514.9887);
    p.bias_gyro_var_inv.setConstant(2950.6680);
    p.if_reject_uwb = true;
    p.if_reject_uwb_in_optimization = false;
    p.reject_uwb_thresh = 0.3;
    p.reject_uwb_window_width = 0.5;
    p.q_nav_uwb_init.setIdentity();
    p.t_nav_uwb_init.setZero();
    fusion.calib_param.offset << std::stod(args.at("lx")), std::stod(args.at("ly")), std::stod(args.at("lz"));
    if (args.at("mode") == "iterate") {
        // An initial iterate is freely optimized, without any added prior factor.
        p.q_nav_uwb_init = Eigen::Quaterniond(std::stod(args.at("qw")), std::stod(args.at("qx")),
                                             std::stod(args.at("qy")), std::stod(args.at("qz"))).normalized();
        p.t_nav_uwb_init << std::stod(args.at("px")), std::stod(args.at("py")), std::stod(args.at("pz"));
    }
    for (const auto& row : readCsv(args.at("anchors"))) {
        auto id = static_cast<uint16_t>(row.at("anchor_id"));
        p.anchor_list[id] = Eigen::Vector3d(row.at("x"), row.at("y"), row.at("z"));
        p.toa_offset[id] = 0;
    }
}

// Reproduce author's run() ordering, and expose only optimized published states.
bool step(SplineFusion& fusion, bool trace) {
    if (!fusion.initialization()) return false;
    if (trace) std::cerr << "initialized; optimizing knots=" << fusion.spline_local.numKnots() << '\n';
    fusion.optimization();
    if (trace) std::cerr << "optimized\n";
    if (fusion.spline_local.numKnots() >= static_cast<size_t>(fusion.window_size)) {
        fusion.window_count++;
        if (fusion.solver_flag == SplineFusion::INITIAL) fusion.solver_flag = SplineFusion::FULLSIZE;
    }
    return true;
}

// Run one fresh process per window/initial iterate to reset author static state.
int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) args[std::string(argv[i]).substr(2)] = argv[i + 1];
        const double duration = std::stod(args.at("duration"));
        SplineFusion fusion;
        configure(fusion, args);
        std::vector<Packet> packets;
        for (const auto& row : readCsv(args.at("imu"))) {
            double t = row.at("t");
            if (t < 0 || t > duration) continue;
            packets.push_back({t, true, {row.at("ax"), row.at("ay"), row.at("az"),
                                        row.at("gx"), row.at("gy"), row.at("gz")}});
        }
        for (const auto& row : readCsv(args.at("radio"))) {
            double t = row.at("t");
            if (t < 0 || t > duration) continue;
            packets.push_back({t, false, {row.at("anchor_id"), row.at("range_m")}});
        }
        std::stable_sort(packets.begin(), packets.end(), [](const Packet& a, const Packet& b) { return a.t < b.t; });
        std::ofstream output(args.at("output"));
        if (!output) throw std::runtime_error("Cannot create output");
        output << "episode,case_id,arrival_s,state_s,x,y,z,wall_s,yaw_rad,status,tag_x,tag_y,tag_z,qw,qx,qy,qz,knots,nav_tx,nav_ty,nav_tz,nav_qw,nav_qx,nav_qy,nav_qz\n";
        output << std::setprecision(16);
        TicToc total;
        size_t next = 0;
        size_t published = 0;
        const int64_t end_ns = static_cast<int64_t>(std::llround(duration * 1e9));
        for (int64_t tick = 0; tick <= end_ns; tick += POLL_NS) {
            while (next < packets.size() && std::llround(packets[next].t * 1e9) <= tick) {
                const auto& packet = packets[next++];
                int64_t stamp = static_cast<int64_t>(std::llround(packet.t * 1e9));
                if (packet.imu) {
                    fusion.imu_buff.emplace_back(stamp,
                        Eigen::Vector3d(packet.values[3], packet.values[4], packet.values[5]),
                        Eigen::Vector3d(packet.values[0], packet.values[1], packet.values[2]));
                } else {
                    fusion.toa_buff.emplace_back(stamp, static_cast<int>(packet.values[0]), packet.values[1]);
                }
            }
            // The numerical core expects both modalities buffered at every growth step.
            if (fusion.imu_buff.empty() || fusion.toa_buff.empty()) continue;
            if (args.count("trace")) std::cerr << "tick=" << tick << '\n';
            if (!step(fusion, args.count("trace") != 0)) continue;
            auto& spline = fusion.spline_local;
            int64_t state_ns = spline.maxTimeNs();
            Eigen::Quaterniond local_q;
            spline.itpQuaternion(state_ns, &local_q);
            const auto& calibration = fusion.calib_param;
            Eigen::Quaterniond global_q = calibration.q_nav_uwb * local_q;
            Eigen::Vector3d global_p = calibration.q_nav_uwb * spline.itpPosition(state_ns) + calibration.t_nav_uwb;
            Eigen::Vector3d tag_p = global_p + global_q * calibration.offset;
            double yaw = std::atan2(2 * (global_q.w() * global_q.z() + global_q.x() * global_q.y()),
                                    1 - 2 * (global_q.y() * global_q.y() + global_q.z() * global_q.z()));
            if (!global_p.allFinite() || !global_q.coeffs().allFinite()) throw std::runtime_error("Nonfinite author estimate");
            output << args.at("episode") << ',' << args.at("case") << ',' << tick * 1e-9 << ',' << state_ns * 1e-9 << ','
                   << global_p.x() << ',' << global_p.y() << ',' << global_p.z() << ',' << total.toc() * 1e-3 << ','
                   << yaw << ",optimized," << tag_p.x() << ',' << tag_p.y() << ',' << tag_p.z() << ','
                   << global_q.w() << ',' << global_q.x() << ',' << global_q.y() << ',' << global_q.z() << ',' << spline.numKnots() << ','
                   << calibration.t_nav_uwb.x() << ',' << calibration.t_nav_uwb.y() << ',' << calibration.t_nav_uwb.z() << ','
                   << calibration.q_nav_uwb.w() << ',' << calibration.q_nav_uwb.x() << ','
                   << calibration.q_nav_uwb.y() << ',' << calibration.q_nav_uwb.z() << '\n';
            published++;
            // Publication precedes removing the oldest knot, exactly as in author run().
            if (fusion.solver_flag == SplineFusion::FULLSIZE) spline.removeOneOldState();
        }
        std::cout << "published=" << published << " wall_s=" << total.toc() * 1e-3 << '\n';
        return published ? 0 : 2;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
