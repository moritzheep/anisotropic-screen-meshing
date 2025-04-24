#pragma once

#include <string>
#include <unordered_map>

class ArgumentParser
{
private:
    std::unordered_map<char, std::string> args_;

public:
    ArgumentParser(int argc, char* argv[])
    {
        for (int i = 1; i != argc; ++i)
        {
            std::string arg(argv[i]);

            if (arg[0] == '-' && i + 1 < argc)
            {
                args_[arg[1]] = std::string(argv[i + 1]);
                // std::cout << arg[1] << ": " << std::string(argv[i + 1]) << "\n";
            }
        }
    }

    bool has_argument(char key) const
    {
        return (args_.count(key) == 1);
    }

    bool has_arguments() const
    {
        return !args_.empty();
    }

    const std::string& get_argument(char key)
    {
        return args_[key];
    }
};