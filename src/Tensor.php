<?php

declare(strict_types=1);

namespace OnnxRuntime;

use EmptyIterator;
use InvalidArgumentException;
use OutOfBoundsException;
use RuntimeException;
use SplFixedArray;
use Traversable;

class Tensor implements TensorInterface
{
    /**
     * Construct a new Tensor instance.
     *
     * @param SplFixedArray $buffer A flat buffer containing tensor data.
     * @param array $shape The shape of the tensor.
     * @param DType $dtype The data type of the tensor.
     */
    public function __construct(
        protected SplFixedArray $buffer,
        protected array         $shape,
        protected DType         $dtype,
    )
    {
    }

    /**
     * Create a Tensor instance from a PHP array.
     *
     * @param array $array The input array.
     * @param DType|null $dtype The data type of the tensor (optional).
     * @param array|null $shape The shape of the tensor (optional).
     * @return static The created Tensor instance.
     * @throws InvalidArgumentException If the shape isn't provided when the array is empty.
     */
    public static function fromArray(array $array, ?DType $dtype = null, ?array $shape = null): static
    {
        if (empty($array) && $shape === null) {
            throw new InvalidArgumentException('Shape must be provided when the array is empty');
        }

        $shape ??= self::generateShape($array);

        $buffer = new SplFixedArray(array_product($shape));

        $index = 0;

        self::flattenArray($array, $buffer, $index, $dtype);

        return new static($buffer, $shape, $dtype);
    }

    /**
     * Create a Tensor instance from a packed binary string.
     *
     * @param string $string The packed binary string containing the tensor data (flat)
     * @param DType $dtype The data type of the tensor.
     * @param array $shape The shape of the tensor.
     * @return static The created Tensor instance.
     * @throws RuntimeException If an error occurs during string unpacking.
     * @throws InvalidArgumentException If the number of elements in the string does not match the shape.
     */
    public static function fromString(string $string, DType $dtype, array $shape): static
    {
        $data = unpack($dtype->packFormat(), $string);

        if ($data === false) {
            throw new RuntimeException('Error unpacking string data');
        }

        if (count($data) != array_product($shape)) {
            throw new InvalidArgumentException('The number of elements in the string does not match the shape');
        }

        $i = 0;

        $buffer = new SplFixedArray(array_product($shape));

        foreach ($data as $value) {
            $buffer[$i++] = $dtype->castValue($value);
        }

        return new static($buffer, $shape, $dtype);
    }

    /**
     * Get the shape of the tensor.
     *
     * @return array The shape of the tensor.
     */
    public function shape(): array
    {
        return $this->shape;
    }

    /**
     * Get the number of dimensions of the tensor.
     *
     * @return int The number of dimensions of the tensor.
     */
    public function ndim(): int
    {
        return count($this->shape);
    }

    /**
     * Get the data type of the tensor.
     *
     * @return DType The data type of the tensor.
     */
    public function dtype(): DType
    {
        return $this->dtype;
    }

    public function buffer(): SplFixedArray
    {
        return $this->buffer;
    }

    public function size(): int
    {
        return array_product($this->shape);
    }

    public function reshape(array $shape): static
    {
        if (array_product($shape) != array_product($this->shape)) {
            throw new InvalidArgumentException('New shape must have the same number of elements');
        }

        return new static($this->buffer, $shape, $this->dtype);
    }

    public function toArray(): array
    {
        $i = 0;
        return self::unflattenArray($this->buffer, $this->shape, $i);
    }

    public function toString(): string
    {
        $string = '';

        foreach ($this->buffer as $value) {
            $string .= pack($this->dtype->packFormat(), $value);
        }

        return $string;
    }

    public function count(): int
    {
        return $this->shape[0];
    }

    public static function generateShape(array $array): array
    {
        $shape = [];

        while (is_array($array)) {
            $shape[] = count($array);
            $array = reset($array);
        }

        return $shape;
    }

    public static function flattenArray($nestedArray, SplFixedArray $buffer, int &$index, ?DType $dtype): void
    {
        foreach ($nestedArray as $value) {
            if (is_array($value)) {
                self::flattenArray($value, $buffer, $index, $dtype);
            } else {
                $dtype ??= DType::inferFrom($value);
                $buffer[$index++] = $dtype->castValue($value);
            }
        }
    }

    public static function unflattenArray(SplFixedArray $buffer, array $shape, int &$index): array
    {
        if (array_product($shape) == 0)
            return [];

        $nestedArray = [];
        $size = array_shift($shape);

        for ($i = 0; $i < $size; $i++) {
            if (empty($shape)) {
                $nestedArray[] = $buffer[$index++];
            } else {
                $nestedArray[] = self::unflattenArray($buffer, $shape, $index);
            }
        }

        return $nestedArray;
    }

    public function getIterator(): Traversable
    {
        if (count($this->shape) == 0)
            return new EmptyIterator();

        $count = $this->count();

        for ($i = 0; $i < $count; $i++) {
            yield $i => $this->offsetGet($i);
        }
    }

    public function offsetExists(mixed $offset): bool
    {
        if (count($this->shape) == 0)
            return false;

        return $offset >= 0 && $offset < $this->shape[0];
    }

    public function offsetGet(mixed $offset): mixed
    {
        if (!$this->offsetExists($offset)) {
            throw new OutOfBoundsException('Index out of bounds');
        }

        $shape = $this->shape;

        if (count($shape) == 1) {
            return $this->buffer[$offset];
        }

        $newShape = array_slice($shape, 1);
        $newSize = array_product($newShape);

        $buffer = new SplFixedArray($newSize);

        for ($i = 0; $i < $newSize; $i++) {
            $buffer[$i] = $this->buffer[$offset * $newSize + $i];
        }

        return new self($buffer, $newShape, $this->dtype);
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        if (!$this->offsetExists($offset)) {
            throw new OutOfBoundsException('Index out of bounds');
        }

        $shape = $this->shape;

        if (!count($shape)) {
            if (!is_scalar($value))
                throw new InvalidArgumentException("Must be scalar type");

            $this->buffer[$offset] = $value;
            return;
        }

        if (!($value instanceof self) || $value->shape() != $shape) {
            throw new InvalidArgumentException('Value must be a tensor with the same shape');
        }

        $buffer = $value->buffer();
        $size = array_product($shape);

        for ($i = 0; $i < $size; $i++) {
            $this->buffer[$offset * $size + $i] = $buffer[$i];
        }
    }

    public function offsetUnset(mixed $offset): void
    {
        throw new RuntimeException('Cannot unset tensor elements');
    }
}