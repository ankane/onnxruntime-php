<?php

use PHPUnit\Framework\TestCase;

use OnnxRuntime\Model;

final class ModelTest extends TestCase
{
    public function testWorks()
    {
        $model = new Model('tests/support/model.onnx');

        $expected = [['name' => 'x', 'type' => 'tensor(float)', 'shape' => [3, 4, 5]]];
        $this->assertEquals($expected, $model->inputs());

        $expected = [['name' => 'y', 'type' => 'tensor(float)', 'shape' => [3, 4, 5]]];
        $this->assertEquals($expected, $model->outputs());

        $x = [[[0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548 ],
               [0.6458941,  0.4375872,  0.891773,   0.96366274, 0.3834415 ],
               [0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606],
               [0.0871293,  0.0202184,  0.83261985, 0.77815676, 0.87001216]],

              [[0.9786183,  0.7991586,  0.46147937, 0.7805292,  0.11827443],
               [0.639921,   0.14335328, 0.9446689,  0.5218483,  0.41466194],
               [0.2645556,  0.7742337,  0.45615032, 0.56843394, 0.0187898 ],
               [0.6176355,  0.6120957,  0.616934,   0.94374806, 0.6818203 ]],

              [[0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667 ],
               [0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076],
               [0.57019675, 0.43860152, 0.9883738,  0.10204481, 0.20887676],
               [0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256 ]]];

        $output = $model->predict(['x' => $x]);
        $this->assertEqualsWithDelta([0.6338603, 0.6715468, 0.6462883, 0.6329476, 0.6043575], $output['y'][0][0], 0.00001);
    }

    public function testInputString()
    {
        $model = new Model('tests/support/identity_string.onnx');
        $x = [['one', 'two'], ['three', 'four']];
        $output = $model->predict(['input:0' => $x]);
        $this->assertEquals($x, $output['output:0']);
    }

    public function testInputBool()
    {
        $model = new Model('tests/support/logical_and.onnx');
        $x = [[false, false], [true, true]];
        $x2 = [[true, false], [true, false]];
        $output = $model->predict(['input:0' => $x, 'input1:0' => $x2]);
        $this->assertEquals([[false, false], [true, false]], $output['output:0']);
    }

    public function testStream()
    {
        $stream = fopen('tests/support/model.onnx', 'rb');
        $model = new Model($stream);
        $expected = [['name' => 'x', 'type' => 'tensor(float)', 'shape' => [3, 4, 5]]];
        $this->assertEquals($expected, $model->inputs());
    }

    public function testLightGBM()
    {
        $model = new Model('tests/support/lightgbm.onnx');

        $expected = [['name' => 'input', 'type' => 'tensor(float)', 'shape' => [1, 2]]];
        $this->assertEquals($expected, $model->inputs());

        $expected = [['name' => 'label', 'type' => 'tensor(int64)', 'shape' => [1]], ['name' => 'probabilities', 'type' => 'seq(map(int64,tensor(float)))', 'shape' => []]];
        $this->assertEquals($expected, $model->outputs());

        $x = [[5.8, 2.8]];

        $output = $model->predict(['input' => $x]);
        $this->assertEquals([1], $output['label']);
        $probabilities = $output['probabilities'][0];
        $this->assertEquals([0, 1, 2], array_keys($probabilities));
        $this->assertEqualsWithDelta([0.2593829035758972, 0.409047931432724, 0.3315691649913788], array_values($probabilities), 0.00001);

        $x2 = [[5.8, 2.8],
               [6.0, 2.2],
               [5.5, 4.2],
               [7.3, 2.9],
               [5.0, 3.4]];

        $labels = [];
        foreach ($x2 as $xi) {
            $output = $model->predict(['input' => [$xi]]);
            $labels[] = $output['label'][0];
        }
        $this->assertEquals([1, 1, 0, 2, 0], $labels);
    }

    public function testRandomForest()
    {
        $model = new Model('tests/support/randomforest.onnx');

        $expected = [['name' => 'float_input', 'type' => 'tensor(float)', 'shape' => [1, 4]]];
        $this->assertEquals($expected, $model->inputs());

        $expected = [
            ['name' => 'output_label', 'type' => 'tensor(int64)', 'shape' => [1]],
            ['name' => 'output_probability', 'type' => 'seq(map(int64,tensor(float)))', 'shape' => []]
        ];
        $this->assertEquals($expected, $model->outputs());

        $x = [[5.8, 2.8, 5.1, 2.4]];

        $output = $model->predict(['float_input' => $x]);
        $this->assertEquals([2], $output['output_label']);
        $probabilities = $output['output_probability'][0];
        $this->assertEquals([0, 1, 2], array_keys($probabilities));
        $this->assertEqualsWithDelta([0.0, 0.0, 1.0000001192092896], array_values($probabilities), 0.00001);
    }

    public function testOutputNames()
    {
        $model = new Model('tests/support/lightgbm.onnx');
        $output = $model->predict(['input' => [[5.8, 2.8]]], outputNames: ['label']);
        $this->assertEquals(['label'], array_keys($output));
    }

    public function testSessionOptions()
    {
        $optimizedPath = tempnam(sys_get_temp_dir(), 'optimized');

        $model = new Model(
            'tests/support/lightgbm.onnx',
            executionMode: OnnxRuntime\ExecutionMode::Sequential,
            graphOptimizationLevel: OnnxRuntime\GraphOptimizationLevel::All,
            interOpNumThreads: 1,
            intraOpNumThreads: 1,
            logSeverityLevel: 4,
            logVerbosityLevel: 4,
            logid: 'test',
            optimizedModelFilepath: $optimizedPath
        );
        $x = [[5.8, 2.8]];
        $model->predict(['input' => $x]);

        $this->assertStringContainsString('onnx', file_get_contents($optimizedPath));
    }

    public function testFreeDimensionOverridesByDenotation()
    {
        $model = new Model('tests/support/abs_free_dimensions.onnx', freeDimensionOverridesByDenotation: ['DATA_BATCH' => 3, 'DATA_CHANNEL' => 5]);
        $this->assertEquals([3, 5, 5], $model->inputs()[0]['shape']);
    }

    public function testFreeDimensionOverridesByName()
    {
        $model = new Model('tests/support/abs_free_dimensions.onnx', freeDimensionOverridesByName: ['Dim1' => 4, 'Dim2' => 6]);
        $this->assertEquals([4, 6, 5], $model->inputs()[0]['shape']);
    }

    public function testInputShapeNames()
    {
        $model = new Model('tests/support/abs_free_dimensions.onnx');
        $this->assertEquals(['Dim1', 'Dim2', 5], $model->inputs()[0]['shape']);
    }

    // TODO improve test
    public function testSessionConfigEntries()
    {
        new Model('tests/support/lightgbm.onnx', sessionConfigEntries: ['key' => 'value']);
        $this->assertTrue(true);
    }

    public function testRunOptions()
    {
        $model = new Model('tests/support/lightgbm.onnx');
        $x = [[5.8, 2.8]];
        $model->predict(['input' => $x], logSeverityLevel: 4, logVerbosityLevel: 4, logid: 'test', terminate: false);
        $this->assertTrue(true);
    }

    public function testInvalidRank()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('Invalid rank for input: x');

        $model = new Model('tests/support/model.onnx');
        $model->predict(['x' => [1]]);
    }

    public function testInvalidDimensions()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('Got invalid dimensions for input: x');

        $model = new Model('tests/support/model.onnx');
        $model->predict(['x' => [[[1]]]]);
    }

    public function testMissingInput()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('No input');

        $model = new Model('tests/support/model.onnx');
        $model->predict([]);
    }

    public function testExtraInput()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('Unknown input: y');

        $model = new Model('tests/support/model.onnx');
        $model->predict(['x' => [1], 'y' => [1]]);
    }

    public function testInvalidOutputName()
    {
        $this->expectException(OnnxRuntime\Exception::class);
        $this->expectExceptionMessage('Invalid output name: bad');

        $model = new Model('tests/support/lightgbm.onnx');
        $x = [[5.8, 2.8]];
        $model->predict(['input' => $x], outputNames: ['bad']);
    }

    public function testMetadata()
    {
        $model = new Model('tests/support/model.onnx');
        $metadata = $model->metadata();
        $this->assertEquals(['hello' => 'world', 'test' => 'value'], $metadata['custom_metadata_map']);
        $this->assertEquals('', $metadata['description']);
        $this->assertEquals('', $metadata['domain']);
        $this->assertEquals('test_sigmoid', $metadata['graph_name']);
        $this->assertEquals('', $metadata['graph_description']);
        $this->assertEquals('backend-test', $metadata['producer_name']);
        $this->assertEquals(9223372036854775807, $metadata['version']);
    }

    public function testLibVersion()
    {
        $this->assertMatchesRegularExpression('/^\d+\.\d+\.\d+$/', OnnxRuntime\FFI::libVersion());
    }
}
