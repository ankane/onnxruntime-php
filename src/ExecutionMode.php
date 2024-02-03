<?php

namespace OnnxRuntime;

enum ExecutionMode: int
{
    case Sequential = 0;
    case Parallel = 1;
}
