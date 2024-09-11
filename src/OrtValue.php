<?php

namespace OnnxRuntime;

class OrtValue
{
    use Utils;

    private $ffi;
    private $api;
    private $allocator;
    private $ptr;
    private $ref;
    private $typeAndShapeInfo;

    public function __construct($ptr, &$ref = null)
    {
        $this->ffi = self::ffi();
        $this->api = self::api();
        $this->allocator = $this->loadAllocator();

        $this->ptr = $ptr;
        $this->ref = $ref;
    }

    public function __destruct()
    {
        ($this->api->ReleaseValue)($this->ptr);
    }

    public static function fromArray($input, ElementType $elementType)
    {
        $ffi = self::ffi();
        $api = self::api();
        $allocator = self::loadAllocator();

        $typeEnum = array_search($elementType, self::typeEnumToElementType());
        if (is_null($typeEnum)) {
            self::unsupportedType('element', $elementType);
        }

        // TODO check shape of each row
        $shape = [];
        $s = $input;
        while (is_array($s)) {
            $shape[] = count($s);
            $s = $s[0];
        }
        $shapeSize = count($shape);
        $inputNodeDims = $ffi->new("int64_t[$shapeSize]");
        for ($i = 0; $i < $shapeSize; $i++) {
            $inputNodeDims[$i] = $shape[$i];
        }

        $ptr = $ffi->new('OrtValue*');
        if ($elementType == ElementType::String) {
            $flatInputSize = array_product($shape);
            $inputTensorValues = $ffi->new("char*[$flatInputSize]");
            $i = 0;
            $strRefs = [];
            self::fillStringTensorValues($input, $inputTensorValues, $shape, $i, $strRefs);

            self::checkStatus(($api->CreateTensorAsOrtValue)($allocator, $inputNodeDims, $shapeSize, $typeEnum, \FFI::addr($ptr)));
            self::checkStatus(($api->FillStringTensor)($ptr, $inputTensorValues, count($inputTensorValues)));
        } else {
            $flatInputSize = array_product($shape);
            $castType = self::castTypes()[$typeEnum];
            $inputTensorValues = $ffi->new("{$castType}[$flatInputSize]");
            $i = 0;
            self::fillTensorValues($input, $inputTensorValues, $shape, $i);

            self::checkStatus(($api->CreateTensorWithDataAsOrtValue)(self::allocatorInfo(), $inputTensorValues, \FFI::sizeof($inputTensorValues), $inputNodeDims, $shapeSize, $typeEnum, \FFI::addr($ptr)));
        }

        return new OrtValue($ptr, $inputTensorValues);
    }

    public static function fromShapeAndType($shape, ElementType $elementType)
    {
        $ffi = self::ffi();
        $api = self::api();
        $allocator = self::loadAllocator();

        $typeEnum = array_search($elementType, self::typeEnumToElementType());
        if (is_null($typeEnum)) {
            self::unsupportedType('element', $elementType);
        }

        $shapeSize = count($shape);
        $inputNodeDims = $ffi->new("int64_t[$shapeSize]");
        for ($i = 0; $i < $shapeSize; $i++) {
            $inputNodeDims[$i] = $shape[$i];
        }

        $ptr = $ffi->new('OrtValue*');
        self::checkStatus(($api->CreateTensorAsOrtValue)($allocator, $inputNodeDims, $shapeSize, $typeEnum, \FFI::addr($ptr)));

        return new OrtValue($ptr);
    }

    private static function fillStringTensorValues($input, $ptr, $shape, &$i, &$refs)
    {
        $dim = array_shift($shape);

        if (count($shape) == 0) {
            for ($j = 0; $j < $dim; $j++) {
                $strPtr = self::cstring($input[$j]);
                $ptr[$i] = $strPtr;
                $refs[] = $strPtr;
                $i++;
            }
        } else {
            for ($j = 0; $j < $dim; $j++) {
                self::fillStringTensorValues($input[$j], $ptr, $shape, $i, $refs);
            }
        }
    }

    private static function fillTensorValues($input, $ptr, $shape, &$i)
    {
        $dim = array_shift($shape);

        if (count($shape) == 0) {
            for ($j = 0; $j < $dim; $j++) {
                $ptr[$i] = $input[$j];
                $i++;
            }
        } else {
            for ($j = 0; $j < $dim; $j++) {
                self::fillTensorValues($input[$j], $ptr, $shape, $i);
            }
        }
    }

    public function isTensor()
    {
        $outType = $this->ffi->new('ONNXType');
        $this->checkStatus(($this->api->GetValueType)($this->ptr, \FFI::addr($outType)));
        return $outType->cdata == $this->ffi->ONNX_TYPE_TENSOR;
    }

    public function dataType()
    {
        $typeinfo = $this->ffi->new('OrtTypeInfo*');
        $this->checkStatus(($this->api->GetTypeInfo)($this->ptr, \FFI::addr($typeinfo)));
        return $this->nodeInfo($typeinfo)['type'];
    }

    public function elementType()
    {
        return $this->typeEnumToElementType()[$this->elementTypeEnum()];
    }

    private function elementTypeEnum()
    {
        return $this->typeAndShapeInfo()[0];
    }

    public function shape()
    {
        return $this->typeAndShapeInfo()[1];
    }

    private function typeAndShapeInfo()
    {
        if (!isset($this->typeAndShapeInfo)) {
            $typeinfo = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
            $this->checkStatus(($this->api->GetTensorTypeAndShape)($this->ptr, \FFI::addr($typeinfo)));
            $this->typeAndShapeInfo = $this->tensorTypeAndShape($typeinfo);

            // TODO use finally
            ($this->api->ReleaseTensorTypeAndShapeInfo)($typeinfo);
        }

        return $this->typeAndShapeInfo;
    }

    public function deviceName()
    {
        return 'cpu';
    }

    public function toObject()
    {
        return $this->createFromOnnxValue($this->ptr);
    }

    public function toPtr()
    {
        return $this->ptr;
    }

    public function dataPtr()
    {
        $castTypes = $this->castTypes();
        $type = $this->elementTypeEnum();
        if (!isset($castTypes[$type])) {
            $this->unsupportedType('element', $type);
        }

        $tensorData = $this->ffi->new($castTypes[$type] . '*');
        $this->checkStatus(($this->api->GetTensorMutableData)($this->ptr, \FFI::addr($tensorData)));
        return $tensorData;
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

    private static $allocatorInfo;

    private static function allocatorInfo()
    {
        if (!isset(self::$allocatorInfo)) {
            self::$allocatorInfo = FFI::instance()->new('OrtMemoryInfo*');
            self::checkStatus((self::api()->CreateCpuMemoryInfo)(1, 0, \FFI::addr(self::$allocatorInfo)));
        }

        return self::$allocatorInfo;
    }
}
