<?php

use PHPUnit\Framework\TestCase;

use OnnxRuntime\ElementType;
use OnnxRuntime\InferenceSession;
use OnnxRuntime\OrtValue;

final class LeakTest extends TestCase
{
    // Test for memory leak in ONNX Runtime handles caused by missing Release* calls
    private const WARMUP = 50;
    private const ITERATIONS = 100;
    private const LIMIT = 5 * 1024 * 1024;
    private const VALGRIND_LIMIT = 20 * 1024 * 1024;

    private const MODEL = 'tests/support/leak_symbolic_dims.onnx';

    public function testNoNativeLeaks()
    {
        if (!function_exists('getrusage') || !isset(getrusage()['ru_maxrss'])) {
            $this->markTestSkipped('getrusage() with ru_maxrss is not available');
        }

        $valgrind = str_contains((string) getenv('LD_PRELOAD'), 'vgpreload');
        $limit = $valgrind ? self::VALGRIND_LIMIT : self::LIMIT;

        $feed = $this->inputFeed();
        for ($i = 0; $i < self::WARMUP; $i++) {
            $this->workload($feed);
        }
        gc_collect_cycles();
        $before = $this->peakRss();

        for ($i = 0; $i < self::ITERATIONS; $i++) {
            $this->workload($feed);
        }
        gc_collect_cycles();
        $growth = $this->peakRss() - $before;

        $this->assertLessThan($limit, $growth, sprintf(
            'Peak RSS grew by %d bytes (%d bytes/iteration) over %d iterations; a native handle is probably not released',
            $growth,
            intdiv($growth, self::ITERATIONS),
            self::ITERATIONS
        ));
    }

    // Test for memory leak in FFI pointer arrays caused by reading elements of anonymous pointer types
    public function testNoHeapGrowth()
    {
        $sess = new InferenceSession('datasets/mul_1.onnx');
        $x = [[1, 2], [3, 4], [5, 6]];
        for ($i = 0; $i < 100; $i++) {
            $sess->run(null, ['X' => $x]);
            OrtValue::fromArray($x, ElementType::Float)->shape();
        }
        gc_collect_cycles();
        $before = memory_get_usage();

        for ($i = 0; $i < 5000; $i++) {
            $sess->run(null, ['X' => $x]);
            OrtValue::fromArray($x, ElementType::Float)->shape();
        }
        gc_collect_cycles();
        $growth = memory_get_usage() - $before;

        $this->assertLessThan(64 * 1024, $growth, sprintf('PHP heap grew by %d bytes over 5000 iterations', $growth));
    }

    private function workload($feed)
    {
        $sess = new InferenceSession(self::MODEL);
        $sess->inputs();
        $sess->outputs();
        $sess->run(null, $feed);

        $values = [];
        foreach ($feed as $name => $input) {
            $values[$name] = OrtValue::fromArray($input, ElementType::Float);
        }
        $output = $sess->runWithOrtValues(null, $values);
        $output[0]->shape();

        $value = OrtValue::fromShapeAndType([256, 256], ElementType::Float);
        $dataPtr = $value->dataPtr();
        for ($i = 0; $i < 256 * 256; $i += 1024) {
            $dataPtr[$i] = 1;
        }
        $value->shape();
        $value->dataType();
    }

    private function inputFeed()
    {
        $sess = new InferenceSession(self::MODEL);
        $feed = [];
        foreach ($sess->inputs() as $input) {
            $x = 1.0;
            foreach ($input['shape'] as $_) {
                $x = [$x];
            }
            $feed[$input['name']] = $x;
        }
        return $feed;
    }

    private function peakRss()
    {
        $maxrss = getrusage()['ru_maxrss'];
        return PHP_OS_FAMILY == 'Darwin' ? $maxrss : $maxrss * 1024;
    }
}
