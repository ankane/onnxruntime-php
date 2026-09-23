<?php

use PHPUnit\Framework\TestCase;

use OnnxRuntime\ElementType;
use OnnxRuntime\InferenceSession;
use OnnxRuntime\OrtValue;

final class InferenceSessionTest extends TestCase
{
    public function testRunWithOrtValues()
    {
        $sess = new InferenceSession('tests/support/lightgbm.onnx');
        $x = OrtValue::fromArray([[5.8, 2.8]], ElementType::Float);
        $output = $sess->runWithOrtValues(null, ['input' => $x]);
        $this->assertTrue($output[0]->isTensor());
        $this->assertEquals('tensor(int64)', $output[0]->dataType());
        $this->assertEquals([1], $output[0]->shape());
        $this->assertEquals([1], $output[0]->toObject());
        $this->assertEquals(1, $output[0]->dataPtr()[0]);
        $this->assertFalse($output[1]->isTensor());
        $this->assertEquals('seq(map(int64,tensor(float)))', $output[1]->dataType());
    }

    public function testRunWithOrtValuesInvalidType()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))');

        $sess = new InferenceSession('tests/support/lightgbm.onnx');
        $x = OrtValue::fromArray([[5.8, 2.8]], ElementType::Double);
        $sess->runWithOrtValues(null, ['input' => $x]);
    }

    public function testRunOrtValueInput()
    {
        $sess = new InferenceSession('tests/support/lightgbm.onnx');
        $x = OrtValue::fromArray([[5.8, 2.8]], ElementType::Float);
        $output = $sess->run(null, ['input' => $x]);
        $this->assertEquals([1], $output[0]);
    }

    public function testProviders()
    {
        $sess = new InferenceSession('tests/support/model.onnx');
        $this->assertContains('CPUExecutionProvider', $sess->providers());
    }

    public function testProvidersCuda()
    {
        // Provider not available: CUDAExecutionProvider
        $sess = new InferenceSession('tests/support/model.onnx', providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']);
        $this->assertNotContains('CUDAExecutionProvider', $sess->providers());
    }

    public function testProvidersCoreML()
    {
        if (PHP_OS_FAMILY != 'Darwin' || php_uname('m') == 'x86_64') {
            $this->markTestSkipped();
        }

        $options = ['providers' => ['CoreMLExecutionProvider', 'CPUExecutionProvider']];
        if (getenv('VERBOSE')) {
            $options['logSeverityLevel'] = 1;
        }
        $sess = new InferenceSession('datasets/mul_1.onnx', ...$options);
        $output = $sess->run(null, ['X' => [[1, 2], [3, 4], [5, 6]]]);
        $this->assertEqualsWithDelta([1, 4], $output[0][0], 0.00001);
        $this->assertEqualsWithDelta([9, 16], $output[0][1], 0.00001);
        $this->assertEqualsWithDelta([25, 36], $output[0][2], 0.00001);
    }

    public function testProfiling()
    {
        $sess = new InferenceSession('tests/support/model.onnx', enableProfiling: true);
        $file = $sess->endProfiling();
        $this->assertStringContainsString('.json', $file);
        unlink($file);
    }

    public function testProfileFilePrefix()
    {
        $sess = new InferenceSession('tests/support/model.onnx', enableProfiling: true, profileFilePrefix: 'hello');
        $file = $sess->endProfiling();
        $this->assertStringContainsString('hello', $file);
        unlink($file);
    }

    public function testTelemetryDisabled()
    {
        // ONNX Runtime skips telemetry when it detects CI, so use a child process that looks like a user machine
        $home = sys_get_temp_dir() . '/onnxruntime-php-' . uniqid();
        mkdir($home);
        $env = array_merge(getenv(), ['HOME' => $home, 'XDG_CACHE_HOME' => "$home/.cache"]);
        foreach (['CI', 'TF_BUILD', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'TRAVIS', 'JENKINS_URL', 'CODEBUILD_BUILD_ID', 'BUILDKITE', 'TEAMCITY_VERSION', 'APPVEYOR', 'BITBUCKET_BUILD_NUMBER', 'SYSTEM_TEAMFOUNDATIONCOLLECTIONURI'] as $name) {
            $env[$name] = 'false';
        }
        unset($env['ORT_DISABLE_TELEMETRY'], $env['ORT_RUNNING_UNIT_TESTS']);
        $code = sprintf('require %s; new OnnxRuntime\InferenceSession(%s);', var_export(__DIR__ . '/../vendor/autoload.php', true), var_export(__DIR__ . '/support/model.onnx', true));
        // FFI may be enabled with -d, which the child would not inherit
        $process = proc_open([PHP_BINARY, '-d', 'extension=ffi', '-d', 'ffi.enable=1', '-r', $code], [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
        $output = stream_get_contents($pipes[1]) . stream_get_contents($pipes[2]);
        $status = proc_close($process);

        $files = [];
        $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($home, FilesystemIterator::SKIP_DOTS), RecursiveIteratorIterator::CHILD_FIRST);
        foreach ($iterator as $file) {
            $files[] = substr($file->getPathname(), strlen($home) + 1);
            $file->isDir() ? rmdir($file->getPathname()) : unlink($file->getPathname());
        }
        rmdir($home);

        $this->assertEquals(0, $status, $output);
        $this->assertEmpty($files, 'ONNX Runtime wrote telemetry files');
    }
}
