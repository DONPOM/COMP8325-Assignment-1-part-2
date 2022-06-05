/**
 * A polyfill
 *
 * <p><b>Require Path:</b> foundation/lib/polyfills/placeholder</p>
 *
 * @module Foundation
 * @submodule Foundation.Polyfills
**/
define('foundation/lib/polyfills/placeholder',[
    'jquery/nyt'
], function ($) {
    if (window.Modernizr.input.placeholder) { return $.noop; }

    var handlePlaceholders = function ($el) {
        var placeholder = '[placeholder]';
        var $inputs = $el ? $el.find(placeholder) : $(placeholder);

         $inputs.each(function () {
            var $input = $(this);
            var $faux;

            if ($input.is('input[type=text]') || $input.is('input[type=password]')) {
                $faux = $('<input type="text" />');
            } else if ($input.is('textarea')) {
                $faux = $('<textarea></textarea>');
            } else {
                return;
            }

            if ($input.is('.withPlaceholder')) {
                $input.hide();
                $input.siblings('.faux')
                    .show()
                    .val($input.attr('placeholder'));

                return true;
            }

            $faux.attr({
                'value': $input.attr('placeholder'),
                'class': $input.attr('class'),
                'style': $input.attr('style')
            }).addClass('faux');

            if ($faux.is('textarea')) {
                $faux.text($input.attr('placeholder'));
            }

            $input
                .addClass('withPlaceholder')
                .after($faux)
                .hide();

            $faux.show().focus(function() {
                $faux.hide();
                $input.show().focus();
            });

            $input.blur(function() {
                if($input.val() === '') {
                    $input.hide();
                    $faux.show();
                }
            });
        });
    };

    return handlePlaceholders;

});
