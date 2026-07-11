#include "csv_logger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

CsvLogger::CsvLogger(const std::string& filePath) {
    const std::filesystem::path path(filePath);

    if (path.has_parent_path()) {
        std::filesystem::create_directories(
            path.parent_path()
        );
    }

    bool writeHeader = true;

    if (std::filesystem::exists(path)) {
        writeHeader =
            std::filesystem::file_size(path) == 0;
    }

    file_.open(filePath, std::ios::app);

    if (file_.is_open() && writeHeader) {
        file_ << "timestamp,fps,width,height\n";
    }
}

bool CsvLogger::isOpen() const {
    return file_.is_open();
}

void CsvLogger::log(
    double fps,
    int width,
    int height
) {
    if (!file_.is_open()) {
        return;
    }

    file_
        << currentTimestamp() << ","
        << std::fixed << std::setprecision(2)
        << fps << ","
        << width << ","
        << height << "\n";

    file_.flush();
}

std::string CsvLogger::currentTimestamp() {
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t nowTime =
        std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};

#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif

    std::ostringstream stream;

    stream << std::put_time(
        &localTime,
        "%Y-%m-%d %H:%M:%S"
    );

    return stream.str();
}

