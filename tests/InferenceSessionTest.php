<?php

use PHPUnit\Framework\TestCase;

final class InferenceSessionTest extends TestCase
{
    public function testProviders()
    {
        $sess = new OnnxRuntime\InferenceSession('tests/support/model.onnx');
        $this->assertContains('CPUExecutionProvider', $sess->providers());
    }

    public function testProvidersCuda()
    {
        // Provider not available: CUDAExecutionProvider
        $sess = new OnnxRuntime\InferenceSession('tests/support/model.onnx', providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']);
        $this->assertNotContains('CUDAExecutionProvider', $sess->providers());
    }

    public function testProvidersCoreML()
    {
        if (PHP_OS_FAMILY != 'Darwin') {
            $this->markTestSkipped();
        }

        $options = ['providers' => ['CoreMLExecutionProvider', 'CPUExecutionProvider']];
        if (getenv('VERBOSE')) {
            $options['logSeverityLevel'] = 1;
        }
        $sess = new OnnxRuntime\InferenceSession('datasets/mul_1.onnx', ...$options);
        $output = $sess->run(null, ['X' => [[1, 2], [3, 4], [5, 6]]]);
        $this->assertEqualsWithDelta([1, 4], $output[0][0], 0.00001);
        $this->assertEqualsWithDelta([9, 16], $output[0][1], 0.00001);
        $this->assertEqualsWithDelta([25, 36], $output[0][2], 0.00001);
    }

    public function testProfiling()
    {
        $sess = new OnnxRuntime\InferenceSession('tests/support/model.onnx', enableProfiling: true);
        $file = $sess->endProfiling();
        $this->assertStringContainsString('.json', $file);
        unlink($file);
    }

    public function testProfileFilePrefix()
    {
        $sess = new OnnxRuntime\InferenceSession('tests/support/model.onnx', enableProfiling: true, profileFilePrefix: 'hello');
        $file = $sess->endProfiling();
        $this->assertStringContainsString('hello', $file);
        unlink($file);
    }
}
