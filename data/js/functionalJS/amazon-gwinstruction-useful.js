var GwInstrumentation = (function() {
    var self = {};

    var cfCount = 0;
    var af = false;

    self.markAF = function() {
        af = true;
        uet('af');
        tryToFireCF();
        P.register('af');
    };

    self.preload = function(src, onload) {
        if(!src) { return; }
        var img = new Image();
        img.onload = function() {
            onload && onload();
        };
        img.src = src;
    };

    self.CF = function() {
        var cf;
        if(typeof arguments[0] === 'string') {
            var src = arguments[0];
            cf = new self.CF();
            self.preload(src, function() {
                cf.ready();
            });
        }
        else if(this.constructor === self.CF) { // invoked with the new keyword
            cf = {};
            cfCount++;
            cf.ready = function() {
                cfCount--;
                tryToFireCF();
            };
            return cf;
        }
        else {
            throw new Error("Incorrect invocation of GwInstrumentation.CF(...)");
        }
    };

    return self;

    function tryToFireCF() {
        if(af && cfCount === 0) {
            uet && uet('cf');
            P && P.register('cf');
        }
    }

}());