
$.ajax({
    url: '//tags.crwdcntrl.net/c/283/cc.js',
    cache: true,
    dataType: 'script',
    success: function() {
        if (typeof LOTCC_283 !== 'undefined') {
            LOTCC_283.add("age", "25");LOTCC_283.add("gen", "M");            LOTCC_283.bcp();
        }
    }
});
