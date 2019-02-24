match($0, /^CLOCK (.*):/, name) { print ARGIND, name[0], $3 }
