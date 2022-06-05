
  P.when('A', 'gw-hover-main-helper').execute(function(A, hh) {
    (function() {
      var $h = A.$('#slideout-hover'),
          hoverWidth = $h.width();

      hh.register('slideout', $h, {
        show: function(e) {
          var $tgt      = A.$(e.target),
              $root     = A.$(e.root),
              tgtPos    = $tgt.offset(),
              rootPos   = $root.offset(),
              tgtRight  = tgtPos.left + $tgt.width(),
              rootRight = rootPos.left + $root.width(),
              bodyWidth = A.$('body').width(),
              left      = tgtRight + hoverWidth <= rootRight ? tgtRight : '',
              right     = left ? '' : bodyWidth - tgtPos.left,
              css       = { 'left'   : left,
                            'right'  : right,
                            'width'  : 0,
                            'top'    : tgtPos.top
                          };

          $h.stop().css(css).show().animate({ width: hoverWidth }, 225);
        },
        hide: function(e) {
          $h.stop().animate({ width: 0 }, 225, function() { $h.hide(); });
        }
      }, { disablePointerEvents: true });
    })();
  });
