<?php
include "really-simple-captcha.php";
$numChars = 7;
$charWidth = 15;
$numImages=100;
// create the instance of the generator
$captcha_instance = new ReallySimpleCaptcha();

// set the generator variables to be what we want
$captcha_instance->char_length = $numChars;
$captcha_instance->font_char_width = $charWidth;
$captcha_instance->img_size = array(($numChars + 1) * $charWidth, 24);

for ($i = 0; $i < $numImages; $i++) {
    // generate a random word
    $word = $captcha_instance->generate_random_word();

    echo ($word . "\n");
    // generate the image file and write it to filesystem
    $captcha_instance->generate_image($word, $word);
}
