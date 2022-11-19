<?php

namespace OnnxRuntime;

// TODO use enum when PHP 8.0 reaches EOL
class GraphOptimizationLevel
{
    public const None = 0;
    public const Basic = 1;
    public const Extended = 2;
    public const All = 99;
}
