<?php

namespace OnnxRuntime;

enum ElementType
{
    case Undefined;
    case Float;
    case UInt8;
    case Int8;
    case UInt16;
    case Int16;
    case Int32;
    case Int64;
    case String;
    case Bool;
    case Float16;
    case Double;
    case UInt32;
    case UInt64;
    case Complex64;
    case Complex128;
    case BFloat16;
}
