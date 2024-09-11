<?php

use PHPUnit\Framework\TestCase;

final class OrtValueTest extends TestCase
{
    public function testOrtvalueFromArray()
    {
        $value = OnnxRuntime\OrtValue::ortvalueFromArray([[5.8, 2.8]], elementType: 'float');
        $this->assertTrue($value->isTensor());
        $this->assertEquals('tensor(float)', $value->dataType());
        $this->assertEquals(1, $value->elementType());
        $this->assertEquals([1, 2], $value->shape());
        $this->assertEquals('cpu', $value->deviceName());
        $this->assertEqualsWithDelta([5.8, 2.8], $value->toObject()[0], 0.00001);
    }
}
