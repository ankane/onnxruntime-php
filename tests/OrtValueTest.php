<?php

use PHPUnit\Framework\TestCase;

use OnnxRuntime\ElementType;
use OnnxRuntime\OrtValue;

final class OrtValueTest extends TestCase
{
    public function testFromArray()
    {
        $value = OrtValue::fromArray([[5.8, 2.8]], ElementType::Float);
        $this->assertTrue($value->isTensor());
        $this->assertEquals('tensor(float)', $value->dataType());
        $this->assertEquals(ElementType::Float, $value->elementType());
        $this->assertEquals([1, 2], $value->shape());
        $this->assertEquals('cpu', $value->deviceName());
        $this->assertEqualsWithDelta([5.8, 2.8], $value->toObject()[0], 0.00001);
        $dataPtr = $value->dataPtr();
        $this->assertEqualsWithDelta(5.8, $dataPtr[0], 0.00001);
        $this->assertEqualsWithDelta(2.8, $dataPtr[1], 0.00001);
    }

    public function testFromShapeAndType()
    {
        $value = OrtValue::fromShapeAndType([1, 2], ElementType::Double);
        $dataPtr = $value->dataPtr();
        $dataPtr[0] = 5.8;
        $dataPtr[1] = 2.8;
        $this->assertTrue($value->isTensor());
        $this->assertEquals('tensor(double)', $value->dataType());
        $this->assertEquals(ElementType::Double, $value->elementType());
        $this->assertEquals([1, 2], $value->shape());
        $this->assertEquals('cpu', $value->deviceName());
        $this->assertEqualsWithDelta([5.8, 2.8], $value->toObject()[0], 0.00001);
    }
}
