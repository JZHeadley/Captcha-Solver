<?php
include "really-simple-captcha.php";
$charWidth = 64;
$fontSize = $charWidth;
$numTrain = 10000;
$numTest = 10000;
$imageHeight = 100;
$trainDir = "../../train";
$testDir = "../../test";

// create the instance of the generator
// set the generator variables to be what we want
function generate_dataset($dir, $numChars, $numImages)
{
    echo ("Generating images in " . $dir . "\n");
    $captcha_instance = new ReallySimpleCaptcha();

    $imageWidth = ($numChars + 1) * $GLOBALS['charWidth'];

    $captcha_instance->tmp_dir = __DIR__ . '/' . $dir;
    if (!file_exists($captcha_instance->tmp_dir)) {
        mkdir($captcha_instance->tmp_dir, 0755, true);
    }
    $imageWidth = ($numChars + 1) * $GLOBALS['charWidth'];
    $captcha_instance->char_length = $numChars;
    $captcha_instance->font_char_width = $GLOBALS['charWidth'];
    $captcha_instance->img_size = array($imageWidth, $GLOBALS['imageHeight']);
    $captcha_instance->font_size = $GLOBALS['fontSize'];
    $captcha_instance->base = array((1 / 12) * $imageWidth, $GLOBALS['imageHeight'] * .75);

    for ($i = 0; $i < $numImages; $i++) {
        // generate a random word
        $word = $captcha_instance->generate_random_word();

        // echo ($word . "\n");
        // generate the image file and write it to filesystem
        $captcha_instance->generate_image($word, $word);
    }
}

generate_dataset($trainDir, 7, $numTrain);
generate_dataset($testDir, 7, $numTest);
