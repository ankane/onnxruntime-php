<?php

use PHPUnit\Framework\TestCase;

use OnnxRuntime\Datasets;
use OnnxRuntime\Model;

final class DatasetsTest extends TestCase
{
    public function testExamples()
    {
        $this->assertExample('logreg_iris.onnx', ['float_input']);
        $this->assertExample('mul_1.onnx', ['X']);
        $this->assertExample('sigmoid.onnx', ['x']);
    }

    public function testBadExample()
    {
        $this->expectException(InvalidArgumentException::class);
        // same message as Python
        $this->expectExceptionMessage("Unable to find example 'bad.onnx'");

        Datasets::example('bad.onnx');
    }

    public function testNoPathTraversal()
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage("Unable to find example '../datasets/sigmoid.onnx'");

        Datasets::example('../datasets/sigmoid.onnx');
    }

    private function assertExample($name, $inputNames)
    {
        $example = Datasets::example($name);
        $model = new Model($example);
        $this->assertEquals($inputNames, array_map(fn ($i) => $i['name'], $model->inputs()));
    }
}
