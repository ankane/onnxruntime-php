<?php

namespace OnnxRuntime;

class Pointer
{
    public $ptr;
    public $free;

    public function __construct($ptr, $free = null)
    {
        $this->ptr = $ptr;
        $this->free = $free;
    }

    public function __destruct()
    {
        if (!is_null($this->free)) {
            ($this->free)($this->ptr);
        }
    }
}
