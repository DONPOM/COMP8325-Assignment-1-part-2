$(document).ready(function() {
    var fnLoadCarousel = function() {
        var iViewportWidth = $(window).width();
        if (iViewportWidth < 768) {
            $.ajax({
                type: "POST",
                url: "/costumes/_home_sliders_xs",
                data: "",
                dataType: "text",
                success: function(sResponse) {
                    $("#carousel").cameraStop();
                    $("#carousel").html(sResponse);
                    $("#carousel").camera({
                        height: '33%',
                        minHeight: '10px',
                        time: 4000,
                        transPeriod: 1000,
                        loaderOpacity: .4,
                        loader: 'pie',
                        piePosition: 'left',
                        navigation: false,
                        playPause: false
                    });
                }
            });
        } else {
            $.ajax({
                type: "POST",
                url: "/costumes/_home_sliders",
                data: "",
                dataType: "text",
                success: function(sResponse) {
                    $("#carousel").cameraStop();
                    $("#carousel").html(sResponse);
                    $("#carousel").camera({
                        height: '26%',
                        minHeight: '10px',
                        time: 4000,
                        transPeriod: 1000,
                        loaderOpacity: .4,
                        loader: 'pie',
                        piePosition: 'left',
                        navigation: false,
                        playPause: false
                    });
                }
            });
        }
    };
    $(window).resize(fnLoadCarousel);
    fnLoadCarousel();
});
