
//<![CDATA[
    // Include Hammer.js on Touch-devices
    (function (j, i, m, d, o) {
        if (!o) return;
        i.getScript(d, function (resp, status, xhr) {
            if ( xhr.readyState === 4 && xhr.status === 200 ) i(j).trigger(m);
        });
    })(
        document,
        window[jimdoData.jQuery],
        'loaded.Hammer',
        '//u.jimdo.com/www62/o/layout/l78ed728bae82f290/js/custom-modules/hammer.js?v=1412258138037',
        ('ontouchstart' in window || (window.DocumentTouch && document instanceof DocumentTouch))
    );