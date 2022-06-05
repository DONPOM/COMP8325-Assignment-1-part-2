
    $('i[dismiss_id]').click(function () {
        $.post("//www.patreon.com/includes/alerts/DismissAlert.php", {
            "dismiss": $(this).attr('dismiss_id'),
            "variant": $(this).attr('variant')
        });
        $(this).closest('.alert_container').slideUp(100);
    })
