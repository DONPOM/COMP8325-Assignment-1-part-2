
                (function($) {
                    if ($('#mainNav2').length) {
                      if ($('#mainNav2').find('a span:not(".cc-invisible")').length === 0) {
                        $('#tp-container').addClass('tp-enable-floating-subnav');
                      } else {
                        $('#tp-container').addClass('tp-enable-subnav');
                      }
                    }
                })(window[jimdoData.jQuery]);
                