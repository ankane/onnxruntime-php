<?php

namespace OnnxRuntime;

class Model
{
    private $session;

    public function __construct($path, ...$sessionOptions)
    {
        $this->session = new InferenceSession($path, ...$sessionOptions);
    }

    public function predict($inputFeed, $outputNames = null, ...$runOptions)
    {
        $outputNames ??= array_map(fn ($o) => $o['name'], $this->outputs());
        $predictions = $this->session->run($outputNames, $inputFeed, ...$runOptions);
        return array_combine($outputNames, $predictions);
    }

    public function inputs()
    {
        return $this->session->inputs();
    }

    public function outputs()
    {
        return $this->session->outputs();
    }

    public function metadata()
    {
        return $this->session->modelmeta();
    }
}
