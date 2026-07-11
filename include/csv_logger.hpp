#ifndef CSV_LOGGER_HPP
#define CSV_LOGGER_HPP

#include <fstream>
#include <string>

class CsvLogger {
public:
    explicit CsvLogger(const std::string& filePath);

    bool isOpen() const;

    void log(
        double fps,
        int width,
        int height
    );

private:
    std::ofstream file_;

    static std::string currentTimestamp();
};

#endif
