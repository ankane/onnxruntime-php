<?php

declare(strict_types=1);

namespace OnnxRuntime;

use ArrayAccess;
use Countable;
use IteratorAggregate;

interface TensorInterface extends ArrayAccess, Countable, IteratorAggregate
{
    public function shape(): array;

    public function ndim(): int;

    public function dtype(): DType;

    public function buffer();

    public function size(): int;

    public static function fromArray(array $array, DType $dtype, array $shape): self;

    public static function fromString(string $string, DType $dtype, array $shape): self;

    public function toArray(): array;

    public function toString(): string;
}