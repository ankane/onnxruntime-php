<?php

use OnnxRuntime\InferenceSession;
use PHPUnit\Framework\TestCase;

final class RunOptionsTest extends TestCase
{
    public function testRunLogSeverityIsConfiguredWithoutVerbosity(): void
    {
        $calls = [];
        $session = $this->sessionWithApiRecorder($calls);

        $this->setRunOptions($session, new stdClass(), 2, null, null, null);

        $this->assertSame([['severity', 2]], $calls);
    }

    public function testRunLogVerbosityIsConfiguredWithoutSeverity(): void
    {
        $calls = [];
        $session = $this->sessionWithApiRecorder($calls);

        $this->setRunOptions($session, new stdClass(), null, 3, null, null);

        $this->assertSame([['verbosity', 3]], $calls);
    }

    private function sessionWithApiRecorder(array &$calls): InferenceSession
    {
        $api = new stdClass();
        $api->RunOptionsSetRunLogSeverityLevel = static function ($options, $level) use (&$calls) {
            $calls[] = ['severity', $level];
        };
        $api->RunOptionsSetRunLogVerbosityLevel = static function ($options, $level) use (&$calls) {
            $calls[] = ['verbosity', $level];
        };

        $session = (new ReflectionClass(InferenceSession::class))->newInstanceWithoutConstructor();
        $apiProperty = new ReflectionProperty(InferenceSession::class, 'api');
        $apiProperty->setAccessible(true);
        $apiProperty->setValue($session, $api);

        return $session;
    }

    private function setRunOptions($session, $options, $severity, $verbosity, $logid, $terminate): void
    {
        $method = new ReflectionMethod(InferenceSession::class, 'setRunOptions');
        $method->setAccessible(true);
        $method->invoke($session, $options, $severity, $verbosity, $logid, $terminate);
    }
}
