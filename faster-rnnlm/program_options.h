#ifndef FASTER_RNNLM_PROGRAM_OPTIONS_H_
#define FASTER_RNNLM_PROGRAM_OPTIONS_H_
#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct IOptionHandler {
  virtual void Parse(const std::string& value) = 0;
  virtual ~IOptionHandler() {}

  virtual const char* GetTypeHint() const = 0;
};


template<class T>
struct OptionHandler : public IOptionHandler {
  explicit OptionHandler(T* var) : var(var) {}

  void Parse(const std::string& value) { std::istringstream(value) >> *var; }

  virtual const char* GetTypeHint() const;

  T* var;
};

template<> void OptionHandler<std::string>::Parse(const std::string& value) { *var = value; }
template<> void OptionHandler<bool>::Parse(const std::string& value) {
  if (value != "0" && value != "1") {
    fprintf(stderr, "Bad bool: %s\n", value.c_str());
    exit(1);
  }
  *var = (value == "1");
}

template<> const char* OptionHandler<std::string>::GetTypeHint() const { return "<string>"; }
template<> const char* OptionHandler<int>::GetTypeHint() const { return "<int>"; }
template<> const char* OptionHandler<uint64_t>::GetTypeHint() const { return "<int>"; }
template<> const char* OptionHandler<float>::GetTypeHint() const { return "<float>"; }
template<> const char* OptionHandler<double>::GetTypeHint() const { return "<float>"; }
template<> const char* OptionHandler<bool>::GetTypeHint() const { return "(0 | 1)"; }


class SimpleOptionParser {
 public:
  ~SimpleOptionParser() {
    std::map<std::string, IOptionHandler*>::iterator it;
    for (it = options_.begin(); it != options_.end(); ++it) {
      delete it->second;
    }
  }

  void Ignore(const std::string& name) {
    ignored_options_.insert(name);
  }

  void AddAlias(const std::string& oldname, const std::string& newname) {
    aliases_[oldname] = newname;
  }

  template<class T>
  void Add(const std::string& name, const std::string& help, T* var) {
    IOptionHandler* handler = new OptionHandler<T>(var);
    options_[name] = handler;
    std::stringstream s;
    s << "  --" << name << " " << handler->GetTypeHint()  << "\n"
      << "    " << help << " (default: " << *var << ")";
    help_lines_.push_back(s.str());
  }

  void Echo(const std::string& text) { help_lines_.push_back(text); }

  void Echo() { Echo(""); }

  void PrintHelp() const {
    for (size_t i = 0; i < help_lines_.size(); ++i) {
      printf("%s\n", help_lines_[i].c_str());
    }
  }

  void Parse(int argc, char** argv) const;

 private:
  std::set<std::string> ignored_options_;
  std::map<std::string, IOptionHandler*> options_;
  std::map<std::string, std::string> aliases_;
  std::vector<std::string> help_lines_;
};


void SimpleOptionParser::Parse(int argc, char** argv) const {
  for (int i = 1; i < argc; i += 2) {
    std::string name = argv[i];
    if (ignored_options_.find(name) != ignored_options_.end()) {
      --i;
      continue;
    }
    if (i + 1 >= argc) {
      fprintf(stderr, "ERROR trailing option without value '%s'\n", name.c_str());
      exit(1);
    }

    size_t trailing_dashes = 0;
    for (; trailing_dashes < name.size() && name[trailing_dashes] == '-'; ++trailing_dashes) {}
    if (trailing_dashes < 1 || trailing_dashes > 2 || name.size() == trailing_dashes) {
      fprintf(stderr, "ERROR: expected argument name at position %d; got %s\n",
          i, name.c_str());
      exit(1);
    }
    name = name.substr(trailing_dashes);

    while (aliases_.find(name) != aliases_.end()) {
      name = aliases_.find(name)->second;
    }

    if (options_.find(name) != options_.end()) {
      options_.find(name)->second->Parse(argv[i + 1]);
    } else {
      fprintf(stderr, "WARNING: unknown option '%s'\n", name.c_str());
    }
  }
}

#endif  // FASTER_RNNLM_PROGRAM_OPTIONS_H_
