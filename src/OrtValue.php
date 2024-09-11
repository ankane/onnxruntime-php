<?php

namespace OnnxRuntime;

class OrtValue
{
    use Utils;

    private $ffi;
    private $api;
    private $ptr;
    private $allocator;

    public function __construct($ptr)
    {
        $this->ffi = FFI::instance();
        $this->api = self::api();

        $this->ptr = $ptr;
        $this->allocator = $this->loadAllocator();
    }

    public function __destruct()
    {
        ($this->api->ReleaseValue)($this->ptr);
    }

    public function toObject()
    {
        return $this->createFromOnnxValue($this->ptr);
    }

    private function createFromOnnxValue($outPtr)
    {
        $outType = $this->ffi->new('ONNXType');
        $this->checkStatus(($this->api->GetValueType)($outPtr, \FFI::addr($outType)));

        if ($outType->cdata == $this->ffi->ONNX_TYPE_TENSOR) {
            $typeinfo = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
            $this->checkStatus(($this->api->GetTensorTypeAndShape)($outPtr, \FFI::addr($typeinfo)));

            [$type, $shape] = $this->tensorTypeAndShape($typeinfo);

            // TODO skip if string
            $tensorData = $this->ffi->new('void*');
            $this->checkStatus(($this->api->GetTensorMutableData)($outPtr, \FFI::addr($tensorData)));

            $outSize = $this->ffi->new('size_t');
            $this->checkStatus(($this->api->GetTensorShapeElementCount)($typeinfo, \FFI::addr($outSize)));
            $outputTensorSize = $outSize->cdata;

            ($this->api->ReleaseTensorTypeAndShapeInfo)($typeinfo);

            $castTypes = $this->castTypes();
            if (isset($castTypes[$type])) {
                $arr = $this->ffi->cast($castTypes[$type] . "[$outputTensorSize]", $tensorData);
            } elseif ($type == $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
                $arr = $this->createStringsFromOnnxValue($outPtr, $outputTensorSize);
            } else {
                $this->unsupportedType('element', $type);
            }

            $i = 0;
            return $this->fillOutput($arr, $shape, $i);
        } elseif ($outType->cdata == $this->ffi->ONNX_TYPE_SEQUENCE) {
            $out = $this->ffi->new('size_t');
            $this->checkStatus(($this->api->GetValueCount)($outPtr, \FFI::addr($out)));

            $ret = [];
            for ($i = 0; $i < $out->cdata; $i++) {
                $seq = $this->ffi->new('OrtValue*');
                $this->checkStatus(($this->api->GetValue)($outPtr, $i, $this->allocator, \FFI::addr($seq)));
                $ret[] = $this->createFromOnnxValue($seq);
            }
            return $ret;
        } elseif ($outType->cdata == $this->ffi->ONNX_TYPE_MAP) {
            $typeShape = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
            $mapKeys = $this->ffi->new('OrtValue*');
            $mapValues = $this->ffi->new('OrtValue*');
            $elemType = $this->ffi->new('ONNXTensorElementDataType');

            $this->checkStatus(($this->api->GetValue)($outPtr, 0, $this->allocator, \FFI::addr($mapKeys)));
            $this->checkStatus(($this->api->GetValue)($outPtr, 1, $this->allocator, \FFI::addr($mapValues)));
            $this->checkStatus(($this->api->GetTensorTypeAndShape)($mapKeys, \FFI::addr($typeShape)));
            $this->checkStatus(($this->api->GetTensorElementType)($typeShape, \FFI::addr($elemType)));

            ($this->api->ReleaseTensorTypeAndShapeInfo)($typeShape);

            // TODO support more types
            if ($elemType->cdata == $this->ffi->ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                $ret = [];
                $keys = $this->createFromOnnxValue($mapKeys);
                $values = $this->createFromOnnxValue($mapValues);
                return array_combine($keys, $values);
            } else {
                $this->unsupportedType('element', $elemType);
            }
        } else {
            $this->unsupportedType('ONNX', $outType->cdata);
        }
    }

    private function fillOutput($ptr, $shape, &$i)
    {
        $dim = array_shift($shape);

        if (count($shape) == 0) {
            $row = [];
            for ($j = 0; $j < $dim; $j++) {
                $row[$j] = $ptr[$i];
                $i++;
            }
            return $row;
        } else {
            $output = [];
            for ($j = 0; $j < $dim; $j++) {
                $output[] = $this->fillOutput($ptr, $shape, $i);
            }
            return $output;
        }
    }

    private function createStringsFromOnnxValue($outPtr, $outputTensorSize)
    {
        $len = $this->ffi->new('size_t');
        $this->checkStatus(($this->api->GetStringTensorDataLength)($outPtr, \FFI::addr($len)));

        $sLen = $len->cdata;
        $s = $this->ffi->new("char[$sLen]");
        $offsets = $this->ffi->new("size_t[$outputTensorSize]");
        $this->checkStatus(($this->api->GetStringTensorContent)($outPtr, $s, $sLen, $offsets, $outputTensorSize));

        $result = [];
        foreach ($offsets as $i => $v) {
            $start = $v;
            $end = $i < $outputTensorSize - 1 ? $offsets[$i + 1] : $sLen;
            $size = $end - $start;
            $result[] = \FFI::string($s + $start, $size);
        }
        return $result;
    }
}
