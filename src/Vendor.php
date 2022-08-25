<?php

namespace OnnxRuntime;

class Vendor
{
    public const VERSION = '1.12.0';

    public const PLATFORMS = [
        'x86_64-linux' => [
            'file' => 'onnxruntime-linux-x64-{{version}}',
            'checksum' => '5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'aarch64-linux' => [
            'file' => 'onnxruntime-linux-aarch64-{{version}}',
            'checksum' => '5820d9f343df73c63b6b2b174a1ff62575032e171c9564bcf92060f46827d0ac',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'x86_64-darwin' => [
            'file' => 'onnxruntime-osx-x86_64-{{version}}',
            'checksum' => '09b17f712f8c6f19bb63da35d508815b443cbb473e16c6192abfaa297c02f600',
            'lib' => 'libonnxruntime.{{version}}.dylib',
            'ext' => 'tgz'
        ],
        'arm64-darwin' => [
            'file' => 'onnxruntime-osx-arm64-{{version}}',
            'checksum' => '23117b6f5d7324d4a7c51184e5f808dd952aec411a6b99a1b6fd1011de06e300',
            'lib' => 'libonnxruntime.{{version}}.dylib',
            'ext' => 'tgz'
        ],
        'x64-windows' => [
            'file' => 'onnxruntime-win-x64-{{version}}',
            'checksum' => '8b5d61204989350b7904ac277f5fbccd3e6736ddbb6ec001e412723d71c9c176',
            'lib' => 'onnxruntime.dll',
            'ext' => 'zip'
        ]
    ];

    public static function check($event)
    {
        $dest = self::defaultLib();
        if (file_exists($dest)) {
            echo "✔ ONNX Runtime found\n";
            return;
        }

        $dir = self::libDir();
        if (!file_exists($dir)) {
            mkdir($dir);
        }

        echo "Downloading ONNX Runtime...\n";

        $file = self::platform('file');
        $ext = self::platform('ext');
        $url = self::withVersion("https://github.com/microsoft/onnxruntime/releases/download/v{{version}}/$file.$ext");
        $contents = file_get_contents($url);

        $checksum = hash('sha256', $contents);
        if ($checksum != self::platform('checksum')) {
            throw new Exception("Bad checksum: $checksum");
        }

        $tempDest = tempnam(sys_get_temp_dir(), 'onnxruntime') . '.' . $ext;
        file_put_contents($tempDest, $contents);

        $archive = new \PharData($tempDest);
        if ($ext != 'zip') {
            $archive = $archive->decompress();
        }
        $archive->extractTo(self::libDir());

        echo "✔ Success\n";
    }

    public static function defaultLib()
    {
        return self::libDir() . '/' . self::libFile();
    }

    private static function libDir()
    {
        return __DIR__ . '/../lib';
    }

    private static function libFile()
    {
        return self::withVersion(self::platform('file') . '/lib/' . self::platform('lib'));
    }

    private static function platform($key)
    {
        return self::PLATFORMS[self::platformKey()][$key];
    }

    private static function platformKey()
    {
        if (PHP_OS_FAMILY == 'Windows') {
            return 'x64-windows';
        } elseif (PHP_OS_FAMILY == 'Darwin') {
            if (php_uname('m') == 'x86_64') {
                return 'x86_64-darwin';
            } else {
                return 'arm64-darwin';
            }
        } else {
            if (php_uname('m') == 'x86_64') {
                return 'x86_64-linux';
            } else {
                return 'aarch64-linux';
            }
        }
    }

    private static function withVersion($str)
    {
        return str_replace('{{version}}', self::VERSION, $str);
    }
}
