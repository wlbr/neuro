.PHONY: demo demo-save demo-load demo-save-gob demo-load-gob tictactoe tictactoe-save tictactoe-load test test-verbose test-race test-coverage fmt vet license-check license-add check ci clean

MODEL_JSON := xor-model.json
MODEL_GOB  := xor-model.gob
TTT_MODEL  := tictactoe-model.json

demo:
	go run ./cmd/xor-demo

demo-save:
	go run ./cmd/xor-demo -format json -save $(MODEL_JSON)

demo-load: $(MODEL_JSON)
	go run ./cmd/xor-demo -format json -load $(MODEL_JSON)

demo-save-gob:
	go run ./cmd/xor-demo -format gob -save $(MODEL_GOB)

demo-load-gob: $(MODEL_GOB)
	go run ./cmd/xor-demo -format gob -load $(MODEL_GOB)

$(MODEL_JSON):
	$(MAKE) demo-save

$(MODEL_GOB):
	$(MAKE) demo-save-gob

tictactoe:
	go run ./cmd/tictactoe-demo

tictactoe-save:
	go run ./cmd/tictactoe-demo -format json -save $(TTT_MODEL)

tictactoe-load: $(TTT_MODEL)
	go run ./cmd/tictactoe-demo -format json -load $(TTT_MODEL)

$(TTT_MODEL):
	$(MAKE) tictactoe-save

test:
	go test ./...

test-verbose:
	go test -v ./...

test-race:
	go test -v -race ./...

test-coverage:
	go test -v -race -coverprofile=coverage.out -covermode=atomic ./...
	@echo "✓ Coverage report generated: coverage.out"

fmt:
	gofmt -w .

vet:
	go vet ./...

license-check:
	@which addlicense > /dev/null || go install github.com/google/addlicense@latest
	addlicense -check -c "Michael Wolber" -l mit ./cmd ./neural

license-add:
	@which addlicense > /dev/null || go install github.com/google/addlicense@latest
	addlicense -c "Michael Wolber" -l mit ./cmd ./neural

check: fmt vet license-check test-race

ci: fmt vet license-check test-race test-coverage

clean:
	rm -f $(MODEL_JSON) $(MODEL_GOB) $(TTT_MODEL) coverage.out coverage.html
