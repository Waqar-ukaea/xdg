#ifndef _XDG_ERROR_H
#define _XDG_ERROR_H

#include <string>
#include <iostream>

#include <fmt/format.h>

// Borrowed from OpenMC
void fatal_error(const std::string& message, int err=-1);

template<typename... Params>
void fatal_error(const std::string& message, const Params&... fmt_args)
{
    fatal_error(fmt::format(message, fmt_args...));
}

#endif