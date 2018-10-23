<?php
include "really-simple-captcha.php";
$numChars = 7;
$charWidth = 64;
$fontSize=$charWidth;
$numImages=5;
$imageHeight=100;
$imageWidth=($numChars + 1) * $charWidth;
$trainDir="train";


// create the instance of the generator
$captcha_instance = new ReallySimpleCaptcha();
// set the generator variables to be what we want
$captcha_instance->tmp_dir = __DIR__ . '/' . $trainDir;

$captcha_instance->char_length = $numChars;
$captcha_instance->font_char_width = $charWidth;
$captcha_instance->img_size = array($imageWidth, $imageHeight);
$captcha_instance->font_size=$fontSize;
$captcha_instance->base=array((1 /12) * $imageWidth, $imageHeight*.75);

for ($i = 0; $i < $numImages; $i++) {
    // generate a random word
    $word = $captcha_instance->generate_random_word();

    echo ($word . "\n");
    // generate the image file and write it to filesystem
    $captcha_instance->generate_image($word, $word);
}
