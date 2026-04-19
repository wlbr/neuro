module github.com/wlbr/neuro

go 1.25

// v1.0.0 and v1.0.1 were published with incorrect module path "ai"
retract (
	v1.0.1
	v1.0.0
)
