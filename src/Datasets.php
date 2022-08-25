<?php

namespace OnnxRuntime;

class Datasets
{
    public static function example($name)
    {
        if (!in_array($name, ['logreg_iris.onnx', 'mul_1.onnx', 'sigmoid.onnx'])) {
            throw new \InvalidArgumentException("Unable to find example '$name'");
        }
        return realpath(__DIR__ . "/../datasets/$name");
    }
}
