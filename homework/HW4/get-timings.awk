match($0, /^CLOCK (.*):/, name) { print ARGIND, name[1], $3 }
