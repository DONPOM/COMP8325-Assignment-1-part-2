/*! Copyright (c) 2010 Brandon Aaron (http://brandonaaron.net)
 * Licensed under the MIT License (LICENSE.txt).
 *
 * Version 2.1.3-pre
 */

(function ($) {

    $.fn.bgiframe1 = (function (s) {
        s = $.extend({
            top: 'auto', // auto == .currentStyle.borderTopWidth
            left: 'auto', // auto == .currentStyle.borderLeftWidth
            width: 'auto', // auto == offsetWidth
            height: 'auto', // auto == offsetHeight
            opacity: true,
            src: 'javascript:false;'
        }, s);
        var html = '<iframe id="bgifr" class="bgiframe" frameborder="0" tabindex="-1" src="' + s.src + '"' +
                           'style="display:block;position:absolute;z-index:-1;' +
                             'top:0' + ';' +
                               'left:0' + ';' +
                               'width:' + (s.width == 'auto' ? (this.offsetWidth + 'px') : prop(s.width)) + ';' +
                               'height:' + (s.height == 'auto' ? (this.offsetHeight + 'px') : prop(s.height)) + ';' +
                               (s.opacity !== false ? 'filter:Alpha(Opacity=\'0\');' : '') +
                              
                        '"/>';
        return this.each(function () {
            $('.bgiframe').remove();
            var $this = $(this);          
            var ifr = document.createElement('iframe');
            ifr.setAttribute("class", 'bgiframe');
            ifr.setAttribute("frameborder", '0');
            ifr.setAttribute("tabindex", '-1');
            ifr.setAttribute("id", 'bgifr');
            //ifr.setAttribute("src", s.src);
            ifr.setAttribute("style", 'display:block;position:absolute;pointer-events: none;z-index:1000;background-color:transparent;' +
                 'top:' + (s.top == 'auto' ? ((parseInt($this.css('borderTopWidth'), 10) || 0) * -1) + 'px' : prop(s.top)) + ';' +
                 'left:' + (s.left == 'auto' ? ((parseInt($this.css('borderLeftWidth'), 10) || 0) * -1) + 'px' : prop(s.left)) + ';' +
                 'width:' + (s.width == 'auto' ? (this.offsetWidth + 'px') : prop(s.width)) + ';' +
                 'height:' + (s.height == 'auto' ? (this.offsetHeight + 'px') : prop(s.height)) + ';' +
                 (s.opacity !== false ? 'filter:Alpha(Opacity=\'0\');' : '') +                           
                + '\'');


            var element = s.element || 'ul.ui-autocomplete';

            if ($.browser.msie && $.browser.version < 9) {
                try {
                    this.insertBefore(document.createElement(html), this.firstChild);
                } catch (ex) { }
            }
            else {
                $(element).before($(ifr));
            }

        });
    });

    // old alias
    $.fn.bgIframe = $.fn.bgiframe1;

    function prop(n) {
        return n && n.constructor === Number ? n + 'px' : n;
    }

})(jQuery);
