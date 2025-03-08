<?php

namespace OnnxRuntime;

class InferenceSession
{
    use Utils;

    private $ffi;
    private $api;
    private $session;
    private $allocator;
    private $inputs;
    private $outputs;

    public function __construct(
        $path,
        $enableCpuMemArena = true,
        $enableMemPattern = true,
        $enableProfiling = false,
        $executionMode = null,
        $freeDimensionOverridesByDenotation = null,
        $freeDimensionOverridesByName = null,
        $graphOptimizationLevel = null,
        $interOpNumThreads = null,
        $intraOpNumThreads = null,
        $logSeverityLevel = null,
        $logVerbosityLevel = null,
        $logid = null,
        $optimizedModelFilepath = null,
        $profileFilePrefix = null,
        $sessionConfigEntries = null,
        $providers = []
    ) {
        $this->ffi = self::ffi();
        $this->api = self::api();

        // session options
        $sessionOptions = $this->ffi->new('OrtSessionOptions*');
        $this->checkStatus(($this->api->CreateSessionOptions)(\FFI::addr($sessionOptions)));
        if ($enableCpuMemArena) {
            $this->checkStatus(($this->api->EnableCpuMemArena)($sessionOptions));
        } else {
            $this->checkStatus(($this->api->DisableCpuMemArena)($sessionOptions));
        }
        if ($enableMemPattern) {
            $this->checkStatus(($this->api->EnableMemPattern)($sessionOptions));
        } else {
            $this->checkStatus(($this->api->DisableMemPattern)($sessionOptions));
        }
        if ($enableProfiling) {
            $this->checkStatus(($this->api->EnableProfiling)($sessionOptions, $this->ortString($profileFilePrefix ?? 'onnxruntime_profile_')));
        } else {
            $this->checkStatus(($this->api->DisableProfiling)($sessionOptions));
        }
        if (!is_null($executionMode)) {
            $this->checkStatus(($this->api->SetSessionExecutionMode)($sessionOptions, $executionMode->value));
        }
        if (!is_null($freeDimensionOverridesByDenotation)) {
            foreach ($freeDimensionOverridesByDenotation as $k => $v) {
                $this->checkStatus(($this->api->AddFreeDimensionOverride)($sessionOptions, $k, $v));
            }
        }
        if (!is_null($freeDimensionOverridesByName)) {
            foreach ($freeDimensionOverridesByName as $k => $v) {
                $this->checkStatus(($this->api->AddFreeDimensionOverrideByName)($sessionOptions, $k, $v));
            }
        }
        if (!is_null($graphOptimizationLevel)) {
            $this->checkStatus(($this->api->SetSessionGraphOptimizationLevel)($sessionOptions, $graphOptimizationLevel->value));
        }
        if (!is_null($interOpNumThreads)) {
            $this->checkStatus(($this->api->SetInterOpNumThreads)($sessionOptions, $interOpNumThreads));
        }
        if (!is_null($intraOpNumThreads)) {
            $this->checkStatus(($this->api->SetIntraOpNumThreads)($sessionOptions, $intraOpNumThreads));
        }
        if (!is_null($logSeverityLevel)) {
            $this->checkStatus(($this->api->SetSessionLogSeverityLevel)($sessionOptions, $logSeverityLevel));
        }
        if (!is_null($logVerbosityLevel)) {
            $this->checkStatus(($this->api->SetSessionLogVerbosityLevel)($sessionOptions, $logVerbosityLevel));
        }
        if (!is_null($logid)) {
            $this->checkStatus(($this->api->SetSessionLogId)($sessionOptions, $logid));
        }
        if (!is_null($optimizedModelFilepath)) {
            $this->checkStatus(($this->api->SetOptimizedModelFilePath)($sessionOptions, $this->ortString($optimizedModelFilepath)));
        }
        if (!is_null($sessionConfigEntries)) {
            foreach ($sessionConfigEntries as $k => $v) {
                $this->checkStatus(($this->api->AddSessionConfigEntry)($sessionOptions, $k, $v));
            }
        }
        foreach ($providers as $provider) {
            if (!in_array($provider, $this->providers())) {
                trigger_error('Provider not available: ' . $provider, E_USER_WARNING);
                continue;
            }

            if ($provider == 'CUDAExecutionProvider') {
                $cudaOptions = $this->ffi->new('OrtCUDAProviderOptionsV2*');
                $this->checkStatus(($this->api->CreateCUDAProviderOptions)(\FFI::addr($cudaOptions)));
                $this->checkStatus(($this->api->SessionOptionsAppendExecutionProvider_CUDA_V2)($sessionOptions, $cudaOptions));
                ($this->api->ReleaseCUDAProviderOptions)($cudaOptions);
            } elseif ($provider == 'CoreMLExecutionProvider') {
                $coremlFlags = 0;
                $this->checkStatus($this->ffi->OrtSessionOptionsAppendExecutionProvider_CoreML($sessionOptions, $coremlFlags));
            } elseif ($provider == 'CPUExecutionProvider') {
                break;
            } else {
                throw new \InvalidArgumentException('Provider not supported: ' . $provider);
            }
        }

        $this->session = $this->loadSession($path, $sessionOptions);
        $this->allocator = $this->loadAllocator();
        $this->inputs = $this->loadInputs();
        $this->outputs = $this->loadOutputs();

        ($this->api->ReleaseSessionOptions)($sessionOptions);
    }

    public function __destruct()
    {
        ($this->api->ReleaseSession)($this->session);
    }

    public function run($outputNames, $inputFeed, $logSeverityLevel = null, $logVerbosityLevel = null, $logid = null, $terminate = null)
    {
        $ortValues = array_combine(array_keys($inputFeed), $this->createInputTensor($inputFeed));
        $output = $this->runWithOrtValues($outputNames, $ortValues, logSeverityLevel: $logSeverityLevel, logVerbosityLevel: $logVerbosityLevel, logid: $logid, terminate: $terminate);
        return array_map(fn ($v) => $v->toObject(), $output);
    }

    public function runWithOrtValues($outputNames, $inputFeed, $logSeverityLevel = null, $logVerbosityLevel = null, $logid = null, $terminate = null)
    {
        $inputFeedSize = count($inputFeed);
        if ($inputFeedSize == 0) {
            throw new Exception('No input');
        }
        $inputTensor = $this->ffi->new("OrtValue*[$inputFeedSize]");
        foreach (array_values($inputFeed) as $i => $ortValue) {
            $inputTensor[$i] = $ortValue->toPtr();
        }

        $outputNames ??= array_map(fn ($v) => $v['name'], $this->outputs);

        $outputsSize = count($outputNames);
        $outputTensor = $this->ffi->new("OrtValue*[$outputsSize]");
        $refs = [];
        $inputNodeNames = $this->createNodeNames(array_keys($inputFeed), $refs);
        $outputNodeNames = $this->createNodeNames($outputNames, $refs);

        // run options
        $runOptions = $this->ffi->new('OrtRunOptions*');
        $this->checkStatus(($this->api->CreateRunOptions)(\FFI::addr($runOptions)));
        if (!is_null($logVerbosityLevel)) {
            $this->checkStatus(($this->api->RunOptionsSetRunLogSeverityLevel)($runOptions, $logSeverityLevel));
        }
        if (!is_null($logVerbosityLevel)) {
            $this->checkStatus(($this->api->RunOptionsSetRunLogVerbosityLevel)($runOptions, $logVerbosityLevel));
        }
        if (!is_null($logid)) {
            $this->checkStatus(($this->api->RunOptionsSetRunTag)($runOptions, $logid));
        }
        if (!is_null($terminate)) {
            if ($terminate) {
                $this->checkStatus(($this->api->RunOptionsSetTerminate)($runOptions));
            } else {
                $this->checkStatus(($this->api->RunOptionsUnsetTerminate)($runOptions));
            }
        }

        $this->checkStatus(($this->api->Run)($this->session, $runOptions, $inputNodeNames, $inputTensor, count($inputFeed), $outputNodeNames, count($outputNames), $outputTensor));

        $output = [];
        foreach ($outputTensor as $t) {
            $output[] = new OrtValue($t);
        }

        // TODO use finally
        ($this->api->ReleaseRunOptions)($runOptions);

        return $output;
    }

    public function inputs()
    {
        return $this->inputs;
    }

    public function outputs()
    {
        return $this->outputs;
    }

    public function modelmeta()
    {
        $keys = $this->ffi->new('char**');
        $numKeys = $this->ffi->new('int64_t');
        $description = $this->ffi->new('char*');
        $domain = $this->ffi->new('char*');
        $graphName = $this->ffi->new('char*');
        $graphDescription = $this->ffi->new('char*');
        $producerName = $this->ffi->new('char*');
        $version = $this->ffi->new('int64_t');

        $metadata = $this->ffi->new('OrtModelMetadata*');
        $this->checkStatus(($this->api->SessionGetModelMetadata)($this->session, \FFI::addr($metadata)));

        $customMetadataMap = [];
        $this->checkStatus(($this->api->ModelMetadataGetCustomMetadataMapKeys)($metadata, $this->allocator, \FFI::addr($keys), \FFI::addr($numKeys)));
        for ($i = 0; $i < $numKeys->cdata; $i++) {
            $keyPtr = $keys[$i];
            $key = \FFI::string($keyPtr);
            $value = $this->ffi->new('char*');
            $this->checkStatus(($this->api->ModelMetadataLookupCustomMetadataMap)($metadata, $this->allocator, $key, \FFI::addr($value)));
            $customMetadataMap[$key] = \FFI::string($value);

            $this->allocatorFree($keyPtr);
            $this->allocatorFree($value);
        }
        $this->allocatorFree($keys);

        $this->checkStatus(($this->api->ModelMetadataGetDescription)($metadata, $this->allocator, \FFI::addr($description)));
        $this->checkStatus(($this->api->ModelMetadataGetDomain)($metadata, $this->allocator, \FFI::addr($domain)));
        $this->checkStatus(($this->api->ModelMetadataGetGraphName)($metadata, $this->allocator, \FFI::addr($graphName)));
        $this->checkStatus(($this->api->ModelMetadataGetGraphDescription)($metadata, $this->allocator, \FFI::addr($graphDescription)));
        $this->checkStatus(($this->api->ModelMetadataGetProducerName)($metadata, $this->allocator, \FFI::addr($producerName)));
        $this->checkStatus(($this->api->ModelMetadataGetVersion)($metadata, \FFI::addr($version)));

        $ret = [
            'custom_metadata_map' => $customMetadataMap,
            'description' => \FFI::string($description),
            'domain' => \FFI::string($domain),
            'graph_name' => \FFI::string($graphName),
            'graph_description' => \FFI::string($graphDescription),
            'producer_name' => \FFI::string($producerName),
            'version' => $version->cdata
        ];

        // TODO use finally
        ($this->api->ReleaseModelMetadata)($metadata);
        $this->allocatorFree($description);
        $this->allocatorFree($domain);
        $this->allocatorFree($graphName);
        $this->allocatorFree($graphDescription);
        $this->allocatorFree($producerName);

        return $ret;
    }

    // return value has double underscore like Python
    public function endProfiling()
    {
        $out = $this->ffi->new('char*');
        $this->checkStatus(($this->api->SessionEndProfiling)($this->session, $this->allocator, \FFI::addr($out)));
        return \FFI::string($out);
    }

    // no way to set providers with C API yet
    // so we can return all available providers
    public function providers()
    {
        $outPtr = $this->ffi->new('char**');
        $lengthPtr = $this->ffi->new('int');
        $this->checkStatus(($this->api->GetAvailableProviders)(\FFI::addr($outPtr), \FFI::addr($lengthPtr)));
        $length = $lengthPtr->cdata;
        $providers = [];
        for ($i = 0; $i < $length; $i++) {
            $providers[] = \FFI::string($outPtr[$i]);
        }
        ($this->api->ReleaseAvailableProviders)($outPtr, $length);
        return $providers;
    }

    private function loadSession($path, $sessionOptions)
    {
        $session = $this->ffi->new('OrtSession*');
        if (is_resource($path) && get_resource_type($path) == 'stream') {
            $contents = stream_get_contents($path);
            $this->checkStatus(($this->api->CreateSessionFromArray)(self::env(), $contents, strlen($contents), $sessionOptions, \FFI::addr($session)));
        } else {
            $this->checkStatus(($this->api->CreateSession)(self::env(), $this->ortString($path), $sessionOptions, \FFI::addr($session)));
        }
        return $session;
    }

    private function loadInputs()
    {
        $inputs = [];
        $numInputNodes = $this->ffi->new('size_t');
        $this->checkStatus(($this->api->SessionGetInputCount)($this->session, \FFI::addr($numInputNodes)));
        for ($i = 0; $i < $numInputNodes->cdata; $i++) {
            $namePtr = $this->ffi->new('char*');
            $this->checkStatus(($this->api->SessionGetInputName)($this->session, $i, $this->allocator, \FFI::addr($namePtr)));
            // freed in nodeInfo
            $typeinfo = $this->ffi->new('OrtTypeInfo*');
            $this->checkStatus(($this->api->SessionGetInputTypeInfo)($this->session, $i, \FFI::addr($typeinfo)));
            $inputs[] = array_merge(['name' => \FFI::string($namePtr)], $this->nodeInfo($typeinfo));
            $this->allocatorFree($namePtr);
        }
        return $inputs;
    }

    private function loadOutputs()
    {
        $outputs = [];
        $numOutputNodes = $this->ffi->new('size_t');
        $this->checkStatus(($this->api->SessionGetOutputCount)($this->session, \FFI::addr($numOutputNodes)));
        for ($i = 0; $i < $numOutputNodes->cdata; $i++) {
            $namePtr = $this->ffi->new('char*');
            $this->checkStatus(($this->api->SessionGetOutputName)($this->session, $i, $this->allocator, \FFI::addr($namePtr)));
            // freed in nodeInfo
            $typeinfo = $this->ffi->new('OrtTypeInfo*');
            $this->checkStatus(($this->api->SessionGetOutputTypeInfo)($this->session, $i, \FFI::addr($typeinfo)));
            $outputs[] = array_merge(['name' => \FFI::string($namePtr)], $this->nodeInfo($typeinfo));
            $this->allocatorFree($namePtr);
        }
        return $outputs;
    }

    private function createInputTensor($inputFeed)
    {
        return array_map(function ($inputName, $input) {
            // TODO support more types
            $inp = null;
            foreach ($this->inputs as $i) {
                if ($i['name'] == $inputName) {
                    $inp = $i;
                    break;
                }
            }
            if (is_null($inp)) {
                throw new Exception("Unknown input: $inputName");
            }

            if ($input instanceof OrtValue) {
                return $input;
            } elseif ($inp['type'] == 'tensor(string)') {
                return OrtValue::fromArray($input, ElementType::String);
            } else {
                $typeEnum = array_search($inp['type'], array_map(fn ($v) => "tensor($v)", $this->elementDataTypes()));
                if ($typeEnum !== false) {
                    return OrtValue::fromArray($input, $this->typeEnumToElementType()[$typeEnum]);
                } else {
                    $this->unsupportedType('input', $inp['type']);
                }
            }
        }, array_keys($inputFeed), array_values($inputFeed));
    }

    private function createNodeNames($names, &$refs)
    {
        $namesSize = count($names);
        $ptr = $this->ffi->new("char*[$namesSize]");
        foreach ($names as $i => $name) {
            $strPtr = $this->cstring($name);
            $ptr[$i] = $strPtr;
            $refs[] = $strPtr;
        }
        return $ptr;
    }

    private function allocatorFree($ptr)
    {
        ($this->api->AllocatorFree)($this->allocator, $ptr);
    }

    // wide string on Windows
    // char string on Linux
    // see ORTCHAR_T in onnxruntime_c_api.h
    private function ortString($str)
    {
        if (PHP_OS_FAMILY == 'Windows') {
            $libc = FFI::libc();
            $max = strlen($str) + 1; // for null byte
            // wchar_t not supported
            // use char instead of casting later
            // since FFI::cast only references data
            $dest = $libc->new('char[' . ($max * 2) . ']');
            $ret = $libc->mbstowcs($dest, $str, $max);
            if ($ret != strlen($str)) {
                throw new Exception('Expected mbstowcs to return ' . strlen($str) . ", got $ret");
            }
            return $dest;
        } else {
            return $str;
        }
    }

    private static $env;

    private static function env()
    {
        // TODO use mutex for thread-safety

        if (!isset(self::$env)) {
            $env = FFI::instance()->new('OrtEnv*');
            (self::api()->CreateEnv)(3, 'Default', \FFI::addr($env));
            register_shutdown_function(fn () => (self::api()->ReleaseEnv)($env));
            // disable telemetry
            // https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md
            self::checkStatus((self::api()->DisableTelemetryEvents)($env));
            self::$env = $env;
        }

        return self::$env;
    }
}
