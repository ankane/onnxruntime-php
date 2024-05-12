<?php

declare(strict_types=1);


namespace OnnxRuntime;

enum DType
{
    case Bool;
    case Int8;
    case Int16;
    case Int32;
    case Int64;
    case Uint8;
    case Uint16;
    case Uint32;
    case Uint64;
    case Float8;
    case Float16;
    case Float32;
    case Float64;
    case String;
    case Complex64;
    case Complex128;
    case BFloat16;


    public function packFormat(): string
    {
        return match ($this) {
            DType::Bool => 'C*',
            DType::Int8 => 'c*',
            DType::Int16 => 's*',
            DType::Int32 => 'l*',
            DType::Int64 => 'q*',
            DType::Uint8 => 'C*',
            DType::Uint16 => 'S*',
            DType::Uint32 => 'L*',
            DType::Uint64 => 'Q*',
            DType::Float8 => 'C*',
            DType::Float16 => 'S*',
            DType::Float32 => 'g*',
            DType::Float64 => 'e*',
            DType::String => 'a*',
        };
    }

    public static function inferFrom(mixed $value): self
    {
        return match (true) {
            is_bool($value) => self::Bool,
            is_int($value) => self::Int32,
            is_float($value) => self::Float64,
            is_string($value) => self::String,
            default => throw new \InvalidArgumentException('Unsupported data type'),
        };
    }

    public function castValue(mixed $value): string|int|bool|float
    {
        return match ($this) {
            DType::Bool => (bool)$value,
            DType::Int8 => (int)$value,
            DType::Int16 => (int)$value,
            DType::Int32 => (int)$value,
            DType::Int64 => (int)$value,
            DType::Uint8 => (int)$value,
            DType::Uint16 => (int)$value,
            DType::Uint32 => (int)$value,
            DType::Uint64 => (int)$value,
            DType::Float8 => (float)$value,
            DType::Float16 => (float)$value,
            DType::Float32 => (float)$value,
            DType::Float64 => (float)$value,
            DType::String => (string)$value,
            self::Complex64,
            self::Complex128,
            self::BFloat16 => throw new \Exception('To be implemented'),
        };
    }
}