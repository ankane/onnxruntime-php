# ONNX Runtime PHP

:fire: [ONNX Runtime](https://github.com/Microsoft/onnxruntime) - the high performance scoring engine for ML models - for PHP

[![Build Status](https://github.com/ankane/onnxruntime-php/workflows/build/badge.svg?branch=master)](https://github.com/ankane/onnxruntime-php/actions)

## Installation

Run:

```sh
composer require ankane/onnxruntime
```

## Getting Started

Load a model and make predictions

```php
$model = new OnnxRuntime\Model('model.onnx');
$model->predict(['x' => [1, 2, 3]]);
```

> Download pre-trained models from the [ONNX Model Zoo](https://github.com/onnx/models)

Get inputs

```php
$model->inputs();
```

Get outputs

```php
$model->outputs();
```

Get metadata

```php
$model->metadata();
```

Get specific outputs

```php
$model->predict(['x' => [1, 2, 3]], outputNames: ['label']);
```

## Session Options

```php
use OnnxRuntime\ExecutionMode;
use OnnxRuntime\GraphOptimizationLevel;

new OnnxRuntime\Model(
    $path,
    enableCpuMemArena: true,
    enableMemPattern: true,
    enableProfiling: false,
    executionMode: ExecutionMode::Sequential, // or Parallel
    freeDimensionOverridesByDenotation: null,
    freeDimensionOverridesByName: null,
    graphOptimizationLevel: GraphOptimizationLevel::None, // or Basic, Extended, All
    interOpNumThreads: null,
    intraOpNumThreads: null,
    logSeverityLevel: 2,
    logVerbosityLevel: 0,
    logid: 'tag',
    optimizedModelFilepath: null,
    profileFilePrefix: 'onnxruntime_profile_',
    sessionConfigEntries: null
);
```

## Run Options

```php
$model->predict(
    $inputFeed,
    outputNames: null,
    logSeverityLevel: 2,
    logVerbosityLevel: 0,
    logid: 'tag',
    terminate: false
);
```

## Inference Session API

You can also use the Inference Session API, which follows the [Python API](https://microsoft.github.io/onnxruntime/python/api_summary.html).

```php
$session = new OnnxRuntime\InferenceSession('model.onnx');
$session->run(null, ['x' => [1, 2, 3]]);
```

The Python example models are included as well.

```php
OnnxRuntime\Datasets::example('sigmoid.onnx');
```

## History

View the [changelog](https://github.com/ankane/onnxruntime-php/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/onnxruntime-php/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/onnxruntime-php/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/onnxruntime-php.git
cd onnxruntime-php
composer install
composer test
```
