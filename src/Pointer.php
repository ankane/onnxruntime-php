<?php

namespace OnnxRuntime;

class Pointer
{
    public $ptr;
    private $free;

    public function __construct($ptr, $free = null)
    {
        $this->ptr = $ptr;
        $this->free = $free;
    }

    public function __destruct()
    {
        if (!is_null($this->free) && !\FFI::isNull($this->ptr)) {
            ($this->free)($this->ptr);
        }
    }

    public function ref()
    {
        return \FFI::addr($this->ptr);
    }

    public function string()
    {
        return \FFI::string($this->ptr);
    }
}
