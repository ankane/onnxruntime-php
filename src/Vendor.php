<?php

namespace OnnxRuntime;

class Vendor
{
    public const VERSION = '1.29.0';

    public const PLATFORMS = [
        'x86_64-linux' => [
            'file' => 'onnxruntime-linux-x64-{{version}}',
            'checksum' => 'c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'aarch64-linux' => [
            'file' => 'onnxruntime-linux-aarch64-{{version}}',
            'checksum' => 'e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f',
            'lib' => 'libonnxruntime.so.{{version}}',
            'ext' => 'tgz'
        ],
        'arm64-darwin' => [
            'file' => 'onnxruntime-osx-arm64-{{version}}',
            'checksum' => 'd0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92',
            'lib' => 'libonnxruntime.{{version}}.dylib',
            'ext' => 'tgz'
        ],
        'x64-windows' => [
            'file' => 'onnxruntime-win-x64-{{version}}',
            'checksum' => 'c9b4b7086b529ad814f428c1bad028e20a25d7dc0699836775faace4ab5b78b2',
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
