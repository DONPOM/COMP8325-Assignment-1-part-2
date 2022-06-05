
    $(document).ready(function () {
        $("#nd-homepage-global-calendar-more").click(function () { document.location.href = Netdania.General.Config.NewsCalendarURL; });

        if (jQuery.browser.mobile) {
            var myScroll;
            function loaded() {
                setTimeout(function () {
                    $("#calendar-container").customScrollbar({
                        skin: "modern-skin",
                        hScroll: false,
                        updateOnWindowResize: true
                    });
                }, 100);
            }
            window.addEventListener('load', loaded, false);
        }
    });
