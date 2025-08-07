local screen = "menu"
local playerChoseEverything = false
local sacrificeStep = 0
local sacrificeTimer = 0

function love.load()
    love.window.setTitle("Simple Menu")
    love.window.setMode(800, 800)
end

function love.draw()
    local ww, wh = love.graphics.getWidth(), love.graphics.getHeight()

    -- Center buttons horizontally
    local buttonW, buttonH = 100, 40
    local buttonStartX = (ww - buttonW) / 2
    local buttonStartY = wh / 2 - 60
    local buttonExitX = buttonStartX
    local buttonExitY = buttonStartY + 60

    local buttonSmallW, buttonSmallH = 120, 40
    local buttonEverythingX = ww/2 - buttonSmallW - 10
    local buttonEverythingY = wh/2 + 20
    local buttonWhatX = ww/2 + 10
    local buttonWhatY = buttonEverythingY

    if screen == "menu" then
        love.graphics.setColor(0.2, 0.6, 1)
        love.graphics.rectangle("fill", buttonStartX, buttonStartY, buttonW, buttonH)
        love.graphics.setColor(1, 1, 1)
        love.graphics.printf("Start", buttonStartX, buttonStartY + 12, buttonW, "center")

        love.graphics.setColor(1, 0.3, 0.3)
        love.graphics.rectangle("fill", buttonExitX, buttonExitY, buttonW, buttonH)
        love.graphics.setColor(1, 1, 1)
        love.graphics.printf("Exit", buttonExitX, buttonExitY + 12, buttonW, "center")
    elseif screen == "sacrifice" then
        love.graphics.clear(0, 0, 0)
        love.graphics.setColor(1, 1, 1)
        love.graphics.printf("what are you willing to sacrifice", 0, wh/2 - 80, ww, "center")
        love.graphics.rectangle("line", buttonEverythingX, buttonEverythingY, buttonSmallW, buttonSmallH)
        love.graphics.printf("everything", buttonEverythingX, buttonEverythingY + 12, buttonSmallW, "center")
        love.graphics.rectangle("line", buttonWhatX, buttonWhatY, buttonSmallW, buttonSmallH)
        love.graphics.printf("what?", buttonWhatX, buttonWhatY + 12, buttonSmallW, "center")
        love.graphics.printf("choose one", 0, wh/2 - 120, ww, "center")
        if playerChoseEverything then
            love.graphics.printf("player has chosen everything", 0, wh/2 - 10, ww, "center")
        end
    elseif screen == "sacrifice_result" then
        love.graphics.clear(0, 0, 0)
        love.graphics.setColor(1, 1, 1)
        if sacrificeStep == 1 then
            love.graphics.printf("player has chosen everything", 0, wh/2 - 10, ww, "center")
        elseif sacrificeStep == 2 then
            love.graphics.printf("you will now be sacrificed", 0, wh/2 - 10, ww, "center")
        elseif sacrificeStep == 3 then
            love.graphics.setColor(1, 0, 0)
            love.graphics.printf("goodbye", 0, wh/2 - 10, ww, "center")
        end
    end

    -- Store button positions for mousepressed
    love.buttonPositions = {
        menu = {
            start = {x = buttonStartX, y = buttonStartY, w = buttonW, h = buttonH},
            exit = {x = buttonExitX, y = buttonExitY, w = buttonW, h = buttonH}
        },
        sacrifice = {
            everything = {x = buttonEverythingX, y = buttonEverythingY, w = buttonSmallW, h = buttonSmallH},
            what = {x = buttonWhatX, y = buttonWhatY, w = buttonSmallW, h = buttonSmallH}
        }
    }
end

function love.update(dt)
    if screen == "sacrifice_result" then
        sacrificeTimer = sacrificeTimer + dt
        if sacrificeStep == 1 and sacrificeTimer > 2 then
            sacrificeStep = 2
            sacrificeTimer = 0
        elseif sacrificeStep == 2 and sacrificeTimer > 2 then
            sacrificeStep = 3
            sacrificeTimer = 0
        elseif sacrificeStep == 3 and sacrificeTimer > 5 then
            love.event.quit()
        end
    end
end

function love.mousepressed(x, y, button)
    if button == 1 then
        local pos = love.buttonPositions
        if screen == "menu" then
            local b = pos.menu
            if x > b.start.x and x < b.start.x + b.start.w and
               y > b.start.y and y < b.start.y + b.start.h then
                screen = "sacrifice"
            end
            if x > b.exit.x and x < b.exit.x + b.exit.w and
               y > b.exit.y and y < b.exit.y + b.exit.h then
                love.event.quit()
            end
        elseif screen == "sacrifice" then
            local b = pos.sacrifice
            if x > b.everything.x and x < b.everything.x + b.everything.w and
               y > b.everything.y and y < b.everything.y + b.everything.h then
                playerChoseEverything = true
                screen = "sacrifice_result"
                sacrificeStep = 1
                sacrificeTimer = 0
            end
            if x > b.what.x and x < b.what.x + b.what.w and
               y > b.what.y and y < b.what.y + b.what.h then
                love.event.quit()
            end
        end
    end
end