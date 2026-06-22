<?php

namespace OnnxRuntime;

class Vendor
{
    public const VERSION = '1.27.0';

    public const PLATFORMS = [
        'x86_64-linux' => [
            'file' => 'onnxruntime-linux-x64-{{version}}',
            'checksum' => '547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'aarch64-linux' => [
            'file' => 'onnxruntime-linux-aarch64-{{version}}',
            'checksum' => '3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'arm64-darwin' => [
            'file' => 'onnxruntime-osx-arm64-{{version}}',
            'checksum' => '545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a',
            'lib' => 'libonnxruntime.{{version}}.dylib',
            'ext' => 'tgz'
        ],
        'x64-windows' => [
            'file' => 'onnxruntime-win-x64-{{version}}',
            'checksum' => 'c5c81710938e68079ff1a192b04897faabe4b43830d48f39f27ecd4e16138bfc',
            'lib' => 'onnxruntime.dll',
            'ext' => 'zip'
        ]
    ];

    public static function check($event = null)
    {
        $dest = self::defaultLib();
        if (file_exists($dest)) {
            echo "✔ ONNX Runtime found\n";
            return;
        }

        if (self::platformKey() == 'x86_64-darwin') {
            echo "✘ ONNX Runtime not found. Run `brew install onnxruntime`\n";
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
        $archive->extractTo(self::libDir(), overwrite: true);

        echo "✔ Success\n";
    }

    public static function defaultLib()
    {
        if (self::platformKey() == 'x86_64-darwin') {
            return '/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib';
        }
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
