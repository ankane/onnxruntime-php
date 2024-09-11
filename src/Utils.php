<?php

namespace OnnxRuntime;

trait Utils
{
    private static function checkStatus($status)
    {
        if (!is_null($status)) {
            $message = (self::api()->GetErrorMessage)($status);
            (self::api()->ReleaseStatus)($status);
            throw new Exception($message);
        }
    }

    private static function api()
    {
        return FFI::api();
    }

    private function unsupportedType($name, $type)
    {
        throw new Exception("Unsupported $name type: $type");
    }

    private function tensorTypeAndShape($tensorInfo)
    {
        $type = $this->ffi->new('ONNXTensorElementDataType');
        $this->checkStatus(($this->api->GetTensorElementType)($tensorInfo, \FFI::addr($type)));

        $numDimsPtr = $this->ffi->new('size_t');
        $this->checkStatus(($this->api->GetDimensionsCount)($tensorInfo, \FFI::addr($numDimsPtr)));
        $numDims = $numDimsPtr->cdata;

        if ($numDims > 0) {
            $nodeDims = $this->ffi->new("int64_t[$numDims]");
            $this->checkStatus(($this->api->GetDimensions)($tensorInfo, $nodeDims, $numDims));
            $dims = $this->readArray($nodeDims);

            $symbolicDims = $this->ffi->new("char*[$numDims]");
            $this->checkStatus(($this->api->GetSymbolicDimensions)($tensorInfo, $symbolicDims, $numDims));
            for ($i = 0; $i < $numDims; $i++) {
                $namedDim = \FFI::string($symbolicDims[$i]);
                if ($namedDim != '') {
                    $dims[$i] = $namedDim;
                }
            }
        } else {
            $dims = [];
        }

        return [$type->cdata, $dims];
    }

    private function nodeInfo($typeinfo)
    {
        $onnxType = $this->ffi->new('ONNXType');
        $this->checkStatus(($this->api->GetOnnxTypeFromTypeInfo)($typeinfo, \FFI::addr($onnxType)));

        if ($onnxType->cdata == $this->ffi->ONNX_TYPE_TENSOR) {
            $tensorInfo = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
            // don't free tensor_info
            $this->checkStatus(($this->api->CastTypeInfoToTensorInfo)($typeinfo, \FFI::addr($tensorInfo)));

            [$type, $shape] = $this->tensorTypeAndShape($tensorInfo);
            $elementDataType = $this->elementDataTypes()[$type];
            return ['type' => "tensor($elementDataType)", 'shape' => $shape];
        } elseif ($onnxType->cdata == $this->ffi->ONNX_TYPE_SEQUENCE) {
            $sequenceTypeInfo = $this->ffi->new('OrtSequenceTypeInfo*');
            $this->checkStatus(($this->api->CastTypeInfoToSequenceTypeInfo)($typeinfo, \FFI::addr($sequenceTypeInfo)));
            $nestedTypeInfo = $this->ffi->new('OrtTypeInfo*');
            $this->checkStatus(($this->api->GetSequenceElementType)($sequenceTypeInfo, \FFI::addr($nestedTypeInfo)));
            $v = $this->nodeInfo($nestedTypeInfo)['type'];

            return ['type' => "seq($v)", 'shape' => []];
        } elseif ($onnxType->cdata == $this->ffi->ONNX_TYPE_MAP) {
            $mapTypeInfo = $this->ffi->new('OrtMapTypeInfo*');
            $this->checkStatus(($this->api->CastTypeInfoToMapTypeInfo)($typeinfo, \FFI::addr($mapTypeInfo)));

            // key
            $keyType = $this->ffi->new('ONNXTensorElementDataType');
            $this->checkStatus(($this->api->GetMapKeyType)($mapTypeInfo, \FFI::addr($keyType)));
            $k = $this->elementDataTypes()[$keyType->cdata];

            // value
            $valueTypeInfo = $this->ffi->new('OrtTypeInfo*');
            $this->checkStatus(($this->api->GetMapValueType)($mapTypeInfo, \FFI::addr($valueTypeInfo)));
            $v = $this->nodeInfo($valueTypeInfo)['type'];

            return ['type' => "map($k,$v)", 'shape' => []];
        } else {
            $this->unsupportedType('ONNX', $onnxType->cdata);
        }
    }

    private function readArray($cdata)
    {
        $arr = [];
        $n = count($cdata);
        for ($i = 0; $i < $n; $i++) {
            $arr[] = $cdata[$i];
        }
        return $arr;
    }

    private function castTypes()
    {
        return [
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => 'float',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => 'uint8_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => 'int8_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => 'uint16_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => 'int16_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => 'int32_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => 'int64_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => 'bool',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => 'double',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => 'uint32_t',
            $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => 'uint64_t'
        ];
    }

    private static $defaultAllocator;

    private static function loadAllocator()
    {
        if (!isset(self::$defaultAllocator)) {
            self::$defaultAllocator = FFI::instance()->new('OrtAllocator*');
            self::checkStatus((self::api()->GetAllocatorWithDefaultOptions)(\FFI::addr(self::$defaultAllocator)));
        }

        return self::$defaultAllocator;
    }
}
