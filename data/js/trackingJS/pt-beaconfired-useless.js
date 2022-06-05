
    Pb.InitEventQueue.addToPageLoad(function(){
        setTimeout(function() {
            var el = $('#liveramp_beacon');
            el.attr('src', el.data('src'));
            Pb.Track.tr("liveramp_beacon_fired");
        }, 1000);
    }, Pb.InitEventQueue.LOW_PRIORITY);
    